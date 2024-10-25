import asyncio
import base64
import logging
import os
from builtins import anext
from typing import Any
from multiprocessing import Queue

from agora.rtc.rtc_connection import RTCConnection, RTCConnInfo
from attr import dataclass

from agora_realtime_ai_api.rtc import Channel, ChatMessage, RtcEngine, RtcOptions

from .logger import setup_logger
from .realtime.struct import InputAudioBufferCommitted, InputAudioBufferSpeechStarted, InputAudioBufferSpeechStopped, InputAudioTranscription, ItemCreated, ItemInputAudioTranscriptionCompleted, RateLimitsUpdated, ResponseAudioDelta, ResponseAudioDone, ResponseAudioTranscriptDelta, ResponseAudioTranscriptDone, ResponseContentPartAdded, ResponseContentPartDone, ResponseCreated, ResponseDone, ResponseOutputItemAdded, ResponseOutputItemDone, ServerVADUpdateParams, SessionUpdate, SessionUpdateParams, SessionUpdated, Voices, to_json, DEFAULT_TURN_DETECTION
from .realtime.connection import RealtimeApiConnection
from .tools import ClientToolCallResponse, ToolContext
from .utils import PCMWriter

# Set up the logger with color and timestamp support
logger = setup_logger(name=__name__, log_level=logging.INFO)

def _monitor_queue_size(queue: asyncio.Queue, queue_name: str, threshold: int = 5) -> None:
    queue_size = queue.qsize()
    if queue_size > threshold:
        logger.warning(f"Queue {queue_name} size exceeded {threshold}: current size {queue_size}")


async def wait_for_remote_user(channel: Channel) -> int:
    remote_users = list(channel.remote_users.keys())
    if len(remote_users) > 0:
        return remote_users[0]

    future = asyncio.Future[int]()

    channel.once("user_joined", lambda conn, user_id: future.set_result(user_id))

    try:
        # Wait for the remote user with a timeout of 30 seconds
        remote_user = await asyncio.wait_for(future, timeout=15.0)
        return remote_user
    except KeyboardInterrupt:
        future.cancel()
        
    except Exception as e:
        logger.error(f"Error waiting for remote user: {e}")
        raise


@dataclass(frozen=True, kw_only=True)
class InferenceConfig:
    system_message: str | None = None
    turn_detection: ServerVADUpdateParams | None = None  # MARK: CHECK!
    voice: Voices | None = None


class RealtimeKitAgent:
    engine: RtcEngine
    channel: Channel
    connection: RealtimeApiConnection
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    command_queue: Queue

    message_queue: asyncio.Queue[ResponseAudioTranscriptDelta] = (
        asyncio.Queue()
    )
    message_done_queue: asyncio.Queue[ResponseAudioTranscriptDone] = (
        asyncio.Queue()
    )
    tools: ToolContext | None = None

    _client_tool_futures: dict[str, asyncio.Future[ClientToolCallResponse]]

    @classmethod
    async def setup_and_run_agent(
        cls,
        *,
        engine: RtcEngine,
        options: RtcOptions,
        inference_config: InferenceConfig,
        tools: ToolContext | None,
        command_queue: Queue,
    ) -> None:
        channel = engine.create_channel(options)
        await channel.connect()

        try:
            async with RealtimeApiConnection(
                base_uri=os.getenv("REALTIME_API_BASE_URI", "wss://api.openai.com"),
                api_key=os.getenv("OPENAI_API_KEY"),
                verbose=False,
            ) as connection:
                session_params=SessionUpdateParams(
                    # MARK: check this
                    turn_detection=inference_config.turn_detection,
                    tools=tools.model_description() if tools else [],
                    tool_choice="auto",
                    input_audio_format="pcm16",
                    output_audio_format="pcm16",
                    instructions=inference_config.system_message,
                    voice=inference_config.voice,
                    model=os.environ.get("OPENAI_MODEL", "gpt-4o-realtime-preview"),
                    modalities=["text", "audio"],
                    temperature=0.8,
                    max_response_output_tokens="inf",
                    input_audio_transcription=InputAudioTranscription(model="whisper-1")
                )
                await connection.send_request(SessionUpdate(session=session_params))

                start_session_message = await anext(connection.listen())
                # assert isinstance(start_session_message, messages.StartSession)
                logger.info(
                    f"Session started: {start_session_message.session.id} model: {start_session_message.session.model}"
                )

                agent = cls(
                    connection=connection,
                    tools=tools,
                    channel=channel,
                    current_session_params=session_params,  # Store session params
                    command_queue=command_queue,
                )
                await agent.run()

        finally:
            await channel.disconnect()
            await connection.close()

    def __init__(
        self,
        *,
        connection: RealtimeApiConnection,
        tools: ToolContext | None,
        channel: Channel,
        current_session_params: SessionUpdateParams,
        command_queue: Queue,
    ) -> None:
        self.connection = connection
        self.tools = tools
        self._client_tool_futures = {}
        self.channel = channel
        self.command_queue = command_queue
        self.current_session_params = current_session_params  # Store the session params
        self.subscribe_user = None
        self.write_pcm = os.environ.get("WRITE_AGENT_PCM", "false") == "true"
        logger.info(f"Write PCM: {self.write_pcm}")

    async def run(self) -> None:
        try:

            def log_exception(t: asyncio.Task[Any]) -> None:
                if not t.cancelled() and t.exception():
                    logger.error(
                        "unhandled exception",
                        exc_info=t.exception(),
                    )

            logger.info("Waiting for remote user to join")
            self.subscribe_user = await wait_for_remote_user(self.channel)
            logger.info(f"Subscribing to user {self.subscribe_user}")
            await self.channel.subscribe_audio(self.subscribe_user)

            async def on_user_left(
                agora_rtc_conn: RTCConnection, user_id: int, reason: int
            ):
                logger.info(f"User left: {user_id}")
                if self.subscribe_user == user_id:
                    self.subscribe_user = None
                    logger.info("Subscribed user left, disconnecting")
                    await self.channel.disconnect()

            self.channel.on("user_left", on_user_left)

            disconnected_future = asyncio.Future[None]()

            def callback(agora_rtc_conn: RTCConnection, conn_info: RTCConnInfo, reason):
                logger.info(f"Connection state changed: {conn_info.state}")
                if conn_info.state == 1:
                    if not disconnected_future.done():
                        disconnected_future.set_result(None)

            self.channel.on("connection_state_changed", callback)

            asyncio.create_task(self.rtc_to_model()).add_done_callback(log_exception)
            asyncio.create_task(self.model_to_rtc()).add_done_callback(log_exception)
            asyncio.create_task(self._commit_audio_loop()).add_done_callback(log_exception)

            asyncio.create_task(self._process_model_messages()).add_done_callback(
                log_exception
            )

            # Task to process commands and disconnection events concurrently
            while True:
                done, pending = await asyncio.wait(
                    [asyncio.create_task(self._check_commands()), disconnected_future],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # If the disconnection future is done, we should stop the loop
                if disconnected_future in done:
                    break

            logger.info("Agent finished running")
        except asyncio.CancelledError:
            logger.info("Agent cancelled")
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise

    # Helper function to check commands in a non-blocking way
    async def _check_commands(self) -> None:
        if not self.command_queue.empty():
            command = self.command_queue.get()  # Use multiprocessing.Queue's blocking get()
            logger.info(f"Got command: {command}")

            if command == "update_instruction":
                new_instruction = self.command_queue.get()  # Get the next item (instruction)
                await self._update_instruction(new_instruction)
            elif command == "update_turn_detection":
                new_turn_detection = self.command_queue.get()  # Get the next item (turn_detection)
                await self._update_turn_detection(new_turn_detection)
            elif command == "send_user_text":
                text = self.command_queue.get()  # Get the next item (text)
                await self._send_user_text(text)
            elif command == "create_response":
                instruction = self.command_queue.get()  # Get the next item (instruction)
                await self._create_response(instruction)
            elif command == "commit_audio_buffer":
                await self._commit_audio_buffer()

        await asyncio.sleep(1)  # Yield control to other tasks

    async def rtc_to_model(self) -> None:
        while self.subscribe_user is None or self.channel.get_audio_frames(self.subscribe_user) is None:
            await asyncio.sleep(0.1)

        audio_frames = self.channel.get_audio_frames(self.subscribe_user)

        # Initialize PCMWriter for receiving audio
        pcm_writer = PCMWriter(prefix="rtc_to_model", write_pcm=self.write_pcm)

        try:
            async for audio_frame in audio_frames:
                # Process received audio (send to model)
                _monitor_queue_size(self.audio_queue, "audio_queue")
                await self.connection.send_audio_data(audio_frame.data)

                # Write PCM data if enabled
                await pcm_writer.write(audio_frame.data)

                await asyncio.sleep(0)  # Yield control to allow other tasks to run

        except asyncio.CancelledError:
            # Write any remaining PCM data before exiting
            await pcm_writer.flush()
            raise  # Re-raise the exception to propagate cancellation

    async def model_to_rtc(self) -> None:
        # Initialize PCMWriter for sending audio
        pcm_writer = PCMWriter(prefix="model_to_rtc", write_pcm=self.write_pcm)

        try:
            while True:
                # Get audio frame from the model output
                frame = await self.audio_queue.get()

                # Process sending audio (to RTC)
                await self.channel.push_audio_frame(frame)

                # Write PCM data if enabled
                await pcm_writer.write(frame)

        except asyncio.CancelledError:
            # Write any remaining PCM data before exiting
            await pcm_writer.flush()
            raise  # Re-raise the cancelled exception to properly exit the task

    async def _update_instruction(self, new_instruction: str) -> None:
        """Update the agent's system instructions and send a SessionUpdate."""
        try:
            # Update only the instructions field of the current session params
            self.current_session_params = self.current_session_params.__class__(
                **{**self.current_session_params.__dict__, "instructions": new_instruction}
            )

            # Send the updated session
            await self.connection.send_request(SessionUpdate(session=self.current_session_params))

            logger.info(f"Session instructions updated to: {new_instruction}")

        except Exception as e:
            logger.error(f"Failed to update agent instructions: {e}")
            raise

    async def _update_turn_detection(self, new_turn_detection: ServerVADUpdateParams|None) -> None:
        """Update the turn detection settings and send a SessionUpdate."""
        try:
            # Update only the turn_detection field of the current session params
            self.current_session_params = self.current_session_params.__class__(
                **{**self.current_session_params.__dict__, "turn_detection": new_turn_detection}
            )

            # Send the updated session
            await self.connection.send_request(SessionUpdate(session=self.current_session_params))

            logger.info(f"Session turn detection updated to: {new_turn_detection}")

        except Exception as e:
            logger.error(f"Failed to update turn detection: {e}")
            raise

    async def _send_user_text(self, text:str) -> None:
        await self.connection.send_user_text(text)
        logger.info(f"User text sent: {text}")

    async def _create_response(self, instruction:str) -> None:
        await self.connection.send_response_create(instruction)
        logger.info(f"Create instruction sent: {instruction}")

    async def _commit_audio_buffer(self) -> None:
        await self.connection.send_audio_data_commit()

    async def _commit_audio_loop(self) -> None:
        """Continuously commit the audio buffer every second while turn detection is None."""
        try:
            while True:
                if self.current_session_params.turn_detection is None:
                    await self._commit_audio_buffer()
                await asyncio.sleep(2)  # Sleep for 2 second between commits
        except asyncio.CancelledError:
            logger.info("Audio commit loop cancelled.")
        except Exception as e:
            logger.error(f"Error in audio commit loop: {e}")
            raise

    async def _process_model_messages(self) -> None:
        async for message in self.connection.listen():
            # logger.info(f"Received message {message=}")
            match message:
                case ResponseAudioDelta():
                    # logger.info("Received audio message")
                    self.audio_queue.put_nowait(base64.b64decode(message.delta))
                    # loop.call_soon_threadsafe(self.audio_queue.put_nowait, base64.b64decode(message.delta))
                    logger.debug(f"TMS:ResponseAudioDelta: response_id:{message.response_id},item_id: {message.item_id}")
                case ResponseAudioTranscriptDelta():
                    # logger.info(f"Received text message {message=}")
                    asyncio.create_task(self.channel.chat.send_message(
                        ChatMessage(
                            message=to_json(message), msg_id=message.item_id
                        )
                    ))

                case ResponseAudioTranscriptDone():
                    logger.info(f"Text message done: {message=}")
                    asyncio.create_task(self.channel.chat.send_message(
                        ChatMessage(
                            message=to_json(message), msg_id=message.item_id
                        )
                    ))
                case InputAudioBufferSpeechStarted():
                    await self.channel.clear_sender_audio_buffer()
                    # clear the audio queue so audio stops playing
                    while not self.audio_queue.empty():
                        self.audio_queue.get_nowait()
                    logger.info(f"TMS:InputAudioBufferSpeechStarted: item_id: {message.item_id}")
                case InputAudioBufferSpeechStopped():
                    logger.info(f"TMS:InputAudioBufferSpeechStopped: item_id: {message.item_id}")
                    pass
                case ItemInputAudioTranscriptionCompleted():
                    logger.info(f"ItemInputAudioTranscriptionCompleted: {message=}")
                    asyncio.create_task(self.channel.chat.send_message(
                        ChatMessage(
                            message=to_json(message), msg_id=message.item_id
                        )
                    ))

                    # Convert transcript to lowercase for case-insensitive comparison
                    transcript_lower = message.transcript.lower()

                    # Check for 'hey' and 'start'
                    if ("hey" in transcript_lower or "okay" in transcript_lower) and ("start" in transcript_lower or "agorastar" in transcript_lower):
                        logger.info("The transcript includes 'Hey' and 'Start'.")
                        await self._update_turn_detection(DEFAULT_TURN_DETECTION)

                    # Check for 'hey' and 'bye-bye'
                    if ("hey" in transcript_lower or "okay" in transcript_lower) and ("bye" in transcript_lower or "stop" in transcript_lower):
                        logger.info("The transcript includes 'Hey' and 'bye'.")
                        await self._update_turn_detection(None)

                #  InputAudioBufferCommitted
                case InputAudioBufferCommitted():
                    logger.info(f"InputAudioBufferCommitted: {message=}")
                    pass
                case ItemCreated():
                    pass
                # ResponseCreated
                case ResponseCreated():
                    pass
                # ResponseDone
                case ResponseDone():
                    pass

                # ResponseOutputItemAdded
                case ResponseOutputItemAdded():
                    pass

                # ResponseContenPartAdded
                case ResponseContentPartAdded():
                    pass
                # ResponseAudioDone
                case ResponseAudioDone():
                    pass
                # ResponseContentPartDone
                case ResponseContentPartDone():
                    pass
                # ResponseOutputItemDone
                case ResponseOutputItemDone():
                    pass
                case SessionUpdated():
                    logger.info(f"SessionUpdated: {message=}")
                    pass
                case RateLimitsUpdated():
                    pass
                case _:
                    logger.warning(f"Unhandled message {message=}")
