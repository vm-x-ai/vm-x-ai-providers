# Contributing to VM-X AI

Welcome to the VM-X AI, we are excited to have you here! We welcome contributions to VM-X AI, and we are happy to help you get started. This document will guide you through the process of contributing to this project.

## Getting Started

To get started, you need to set up your development environment.

### Prerequisites

- [Node.js 20.x](https://nodejs.org/en/download/)
- [PNPM 8.x](https://pnpm.io/installation)

**IMPORTANT**: The PNPM version needs to be 8.x. You can check the version by running `pnpm -v`.

### Installation

```bash
pnpm install
```

### Build all packages

```bash
pnpm nx run-many --target=build --all
```

or only the affected packages

```bash
pnpm nx affected:build
```

### Lint all packages

```bash
pnpm nx run-many --target=lint --all
```

or only the affected packages

```bash
pnpm nx affected:lint
```

### Test all packages

```bash
pnpm nx run-many --target=test --all
```

or only the affected packages

```bash
pnpm nx affected:test
```

## Committing Changes

### Commit message linting

This workspace uses [commitlint](https://commitlint.js.org/) to enforce a consistent commit message format. The commit message format is based on the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

Example:

```text
feat: my awesome change
```

More details here: [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)

### Commit message prompt

You can also use the `pnpm run commit` command that will prompt you all the necessary information to create a commit message following the Conventional Commits specification.

**NOTE**: The `git commit -m ...` command still works as usual.
