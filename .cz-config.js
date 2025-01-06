module.exports = {
  types: [
    { value: 'feat', name: '✨ feat:\tA new feature' },
    { value: 'fix', name: '🐛 fix:\tA bug fix' },
    { value: 'docs', name: '📚 docs:\tDocumentation only changes' },
    { value: 'chore', name: "🔧 chore:\tOther changes that don't modify src or test files" },
    {
      value: 'style',
      name: '💎 style:\tChanges that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)',
    },
    {
      value: 'refactor',
      name: '📦 refactor:\tA code change that neither fixes a bug nor adds a feature',
    },
    {
      value: 'perf',
      name: '🚀 perf:\tA code change that improves performance',
    },
    {
      value: 'test',
      name: '🚨 test:\tAdding missing tests or correcting existing tests',
    },
    {
      value: 'build',
      name: '🛠  build:\tChanges that affect the build system or external dependencies (example scopes: nx, npm)',
    },
    {
      value: 'ci',
      name: '🤖 ci:\tChanges to our CI configuration files and scripts',
    },
  ],

  messages: {
    type: "Select the type of change that you're committing:",
    customScope: 'What is the scope of this change (e.g. component or project name) (optional):',
    subject: 'Write a short, imperative present tense description of the change:\n',
    body: 'Provide a longer description of the change (optional). Use "|" to break new line:\n',
    confirmCommit: 'Are you sure you want to proceed with the commit above?',
  },

  allowCustomScopes: true,
  skipQuestions: ['footer'],

  subjectLimit: 100,
  footerPrefix: '',
  allowBreakingChanges: ['feat', 'fix'],
};
