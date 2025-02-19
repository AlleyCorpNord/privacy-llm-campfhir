# Campfhir

## Development Pre-requisites

- [Visual Studio Code](https://code.visualstudio.com/Download)
- Docker
  - Please make sure that you are using Docker Compose V2 (look in your docker
    desktop settings for it)
- [Remote Development extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)

## Get Started

1. In case you are unfamiliar with Remote Development extension, in VS Code,
   open the campfhir code folder and an option to 'Folder contains a Dev
   Container configuration file. Reopen folder to develop in a container' will
   pop up on the right bottom. Select 'Reopen in container'. If this is the
   first time building the project, using the command tools you can Shift +
   Cmd + P and select 'Dev containers: rebuild containers without cache`.
2. Then you can start querying the API at `http://localhost:8889`

---

## Usage

### Start the webserver project:

This will watch the project directory and restart as necessary.

```
yarn chat
```

### Start the UI component:

```
yarn chat-styles
```

### Format the deno TS sourcecode:

```
yarn lint
```

### Clear the deno package cache:

```
yarn clear-cache
```

### Devcontainer CLI

Sometimes, it's more convenient to run the code directly in the computer's directory instead of within the `devcontainer`.
This can be achieved by [installing](https://code.visualstudio.com/docs/devcontainers/devcontainer-cli) and running the `devcontainer` CLI.

Here is a sample command that can be used to run the `devcontainer` through the CLI:

```shell
devcontainer up --id-label name="Campfhir" --workspace-folder .
```

**Note: Hot refresh will only work if run using the `devcontainer` CLI**
