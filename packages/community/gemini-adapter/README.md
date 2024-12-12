# Gemini Adapter AI Provider

## Requirements

- VM-X AI Account. [sign up here](https://console.vm-x.ai/)
  - VM-X Personal Access Token. [generate here](https://console.vm-x.ai/account/profile)

## Usage

### Build

VM-X AI Provider uses ESBuild to bundle the code, the following command will generate a bundle in the `dist` folder.

```bash
nx build gemini-adapter
```

### Debug

Connect your local VM-X AI Provider with the VM-X Console and test your changes in real-time.

```bash
nx run gemini-adapter:watch
```

**NOTE**: You need a VM-X Workspace and environment to test your changes, [create one here](https://console.vm-x.ai/getting-started), if you already have them, you can find the `Workspace ID` and `Environment ID` in the URL of the VM-X Console.

**TIP**: Add the `VMX_WORKSPACE_ID`, `VMX_ENVIRONMENT_ID` and `VMX_PAT` to the `.env.local` file in the root of the project, to avoid keep passing them as arguments every time you run the `watch` command.

### Test

```bash
nx test gemini-adapter
```

### Publish

```bash
nx run gemini-adapter:publish
```

**TIP**: Add the `VMX_PAT` to the `.env.local` file in the root of the project, to avoid keep passing them as arguments every time you run the `publish` command.
