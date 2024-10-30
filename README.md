# Realtime Agent

This project demonstrates how to deliver ultra-low latency access to OpenAI with exceptional audio quality using Agora's SD-RTN and OpenAI's Realtime API. By integrating Agora's SDK with OpenAI's Realtime API, it ensures seamless performance and minimal delay across the globe.

## Prerequisites

Before running the demo, ensure you have the following installed and configured:

- Python 3.11 or above

- Agora account:

  - [Login to Agora](https://console.agora.io/en/)
  - Create a [New Project](https://console.agora.io/projects), using `Secured mode: APP ID + Token` to obtain an App ID and App Certificate.

- OpenAI account:

  - [Login to OpenAI](https://platform.openai.com/signup)
  - Go to Dashboard and [obtain your API key](https://platform.openai.com/api-keys).

- Additional Packages:

  - On macOS:
    ```bash
    brew install ffmpeg portaudio
    ```
  - On Ubuntu (verified on versions 22.04 & 24.04):
    ```bash
    sudo apt install portaudio19-dev python3-dev build-essential
    sudo apt install ffmpeg
    ```

## Network Architecture

<!-- <img src="./architecture.png" alt="architecture" width="700" height="400" /> -->
<picture>
  <source srcset="architecture-dark-theme.png" media="(prefers-color-scheme: dark)">
  <img src="architecture-light-theme.png" alt="Architecture diagram of Conversational Ai by Agora and OpenAi">
</picture>

## Organization of this Repo

- `realtimeAgent/realtime` contains the Python implementation for interacting with the Realtime API.
- `realtimeAgent/agent.py` includes a demo agent that leverages the `realtime` module and the [agora-realtime-ai-api](https://pypi.org/project/agora-realtime-ai-api/) package to build a simple application.
- `realtimeAgent/main.py` provides a web server that allows clients to start and stop AI-driven agents.

## Run the Demo

### Setup and run the backend

1. Create a `.env` file for the backend. Copy `.env.example` to `.env` in the root of the repo and fill in the required values:
   ```bash
   cp .env.example .env
   ```
1. Create a virtual environment:
   ```bash
   python3 -m venv venv && source venv/bin/activate
   ```
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
1. Run the demo agent:
   ```bash
   python -m realtime_agent.main agent --channel_name=<channel_name> --uid=<agent_uid>
   ```

### Start HTTP Server

1. Run the http server to start demo agent via restful service
   ```bash
   python -m realtime_agent.main server
   ```
   The server provides a simple layer for managing agent processes.

### API Resources

- [POST /start](#post-start)
- [POST /stop](#post-stop)
- [POST /control_agent](#control_agent)

### POST /start

This api starts an agent with given graph and override properties. The started agent will join into the specified channel, and subscribe to the uid which your browser/device's rtc use to join.

| Param        | Description                                                                                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| channel_name | (string) channel name, it needs to be the same with the one your browser/device joins, agent needs to stay with your browser/device in the same channel to communicate |
| uid          | (int)the uid which ai agent use to join                                                                                                                                |
| system_instruction    | The system instruction for the agent                                                                                                                          |
| voice        | The voice of the agent                                                                                                                                                 |

Example:

```bash
curl 'http://localhost:8080/start_agent' \
  -H 'Content-Type: application/json' \
  --data-raw '{
    "channel_name": "test",
    "uid": 123
  }'
```

### POST /stop

This api stops the agent you started

| Param        | Description                                                |
| ------------ | ---------------------------------------------------------- |
| channel_name | (string) channel name, the one you used to start the agent |

Example:

```bash
curl 'http://localhost:8080/stop_agent' \
  -H 'Content-Type: application/json' \
  --data-raw '{
    "channel_name": "test"
  }'
```

### POST /control_agent

This API controls an already running agent by sending various commands to adjust the agent's behavior or update its settings. Each command has different parameters, allowing you to update instructions, turn detection, or send user text for the agent to process. The agent must be running in the specified channel for the command to be executed.

| Param             | Type     | Description                                                                                                                                                                    |
|-------------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `channel_name`    | `string` | (Required) The name of the channel where the agent is running. It must match the active channel of the agent that the command targets.                                        |
| `command`         | `string` | (Required) The command to be sent to the agent. Possible values are: `update_instruction`, `update_turn_detection`, `send_user_text`, `create_response`, and `commit_audio_buffer`. |
| `new_instruction` | `string` | (Optional) The new system instruction for the agent. Required when `command` is `update_instruction` or `create_response`.                                                     |
| `new_turn_detection` | `boolean` | (Optional) Specifies whether to enable or disable turn detection. When `command` is `update_turn_detection`, pass `true` to enable or `false` to disable.                    |
| `input_text`       | `string` | (Optional) Text message from the user to be processed by the agent. Required when `command` is `send_user_text` or `send_system_text`.                                                              |

#### Commands

Each command controls a different aspect of the agent’s operation. Below are the details of each command and its requirements.

##### `update_instruction`
Updates the agent's current system instruction with a new one.

- **Required Fields**: `channel_name`, `command`, `new_instruction`
- **Description**: Updates the system instruction for the agent, which adjusts its behavior or response style. The new instruction is specified in `new_instruction`.

##### `update_turn_detection`
Enables or disables turn detection based on the `new_turn_detection` value.

- **Required Fields**: `channel_name`, `command`, `new_turn_detection`
- **Description**: Sets the turn detection to `DEFAULT_TURN_DETECTION` if `new_turn_detection` is `true`, or disables it by setting to `None` if `new_turn_detection` is `false`.

##### `send_system_text`
Sends a message as if it came from a system, allowing the agent to process it as input.

- **Required Fields**: `channel_name`, `command`, `input_text`
- **Description**: Sends the specified `input_text` as input for the agent to respond to or use as part of its processing.

##### `send_user_text`
Sends a message as if it came from a user, allowing the agent to process it as input.

- **Required Fields**: `channel_name`, `command`, `input_text`
- **Description**: Sends the specified `input_text` as input for the agent to respond to or use as part of its processing.

##### `create_response`
Generates a response based on the `new_instruction` provided.

- **Required Fields**: `channel_name`, `command`, `new_instruction`
- **Description**: Instructs the agent to generate a response according to the provided `new_instruction`. This is used to create specific responses directly.

##### `commit_audio_buffer`
Commits the current audio buffer, processing it for output or storage.

- **Required Fields**: `channel_name`, `command`
- **Description**: Instructs the agent to commit its current audio buffer. This command is useful for audio processing or storage purposes.

#### Example Request

```bash
curl -X POST 'http://localhost:8080/control_agent' \
-H 'Content-Type: application/json' \
--data-raw '{
    "channel_name": "test",
    "command": "update_instruction",
    "new_instruction": "You are now assisting Yohei with game suggestions."
}'
```

#### Responses

- **Success**:
  - Status `200 OK`
  - JSON: `{ "status": "Command <command> sent to agent." }`

### Voice-Triggered Agent Control

This feature enables the agent to start or stop based on specific user voice commands, allowing hands-free control.

- Saying `Hey...Start` will resume a paused agent, allowing it to continue operation within the specified channel.
- Saying `Hey...Stop` will pause the currently running agent, effectively stopping its interaction until resumed.

This functionality enables more natural, responsive interactions, where users can start or stop the agent dynamically based on their spoken commands.

### Front-End for Testing

To test agents, use Agora's [Voice Call Demo](https://webdemo.agora.io/basicVoiceCall/index.html).
