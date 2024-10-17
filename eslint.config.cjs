const { FlatCompat } = require('@eslint/eslintrc');
const nxEslintPlugin = require('@nx/eslint-plugin');
const eslintPluginVitest = require('eslint-plugin-vitest');
const eslintPluginImport = require('eslint-plugin-import');
const js = require('@eslint/js');

const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
});

module.exports = [
  {
    plugins: {
      '@nx': nxEslintPlugin,
      vitest: eslintPluginVitest,
      import: eslintPluginImport,
    },
  },
  {
    languageOptions: {
      parserOptions: { ecmaVersion: 'latest' },
      globals: {
        ...eslintPluginVitest.environments.env.globals,
      },
    },
  },
  {
    files: ['**/*.ts', '**/*.tsx', '**/*.js', '**/*.jsx'],
    ignores: ['**/eslint.config.js'],
    rules: {
      '@nx/enforce-module-boundaries': [
        'error',
        {
          allowCircularSelfDependency: true,
          enforceBuildableLibDependency: true,
          allow: [],
          depConstraints: [
            {
              sourceTag: '*',
              onlyDependOnLibsWithTags: ['*'],
            },
          ],
        },
      ],
    },
  },
  {
    files: ['**/*.ts', '**/*.tsx', '**/*.js', '**/*.jsx'],
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          ignoreRestSiblings: true,
          argsIgnorePattern: '^_',
        },
      ],
      '@typescript-eslint/consistent-type-imports': ['error', { disallowTypeAnnotations: false }],
      'import/order': [
        'error',
        {
          alphabetize: {
            order: 'asc',
            caseInsensitive: true,
          },
        },
      ],
    },
  },
  ...compat.config({ extends: ['plugin:@nx/typescript'] }).map((config) => ({
    ...config,
    files: ['**/*.ts', '**/*.tsx'],
    rules: {
      ...config.rules,
      '@typescript-eslint/no-extra-semi': 'error',
      'no-extra-semi': 'off',
    },
  })),
  ...compat.config({ extends: ['plugin:@nx/javascript'] }).map((config) => ({
    ...config,
    files: ['**/*.js', '**/*.jsx'],
    rules: {
      ...config.rules,
      '@typescript-eslint/no-extra-semi': 'error',
      'no-extra-semi': 'off',
    },
  })),
  {
    files: ['**/*.spec.ts', '**/*.spec.tsx', '**/*.spec.js', '**/*.spec.jsx'],
    rules: {
      ...eslintPluginVitest.configs.recommended.rules,
    },
  },
  { ignores: ['**/.next', '**/.nextjs-cdk', '**/cdk.out', '**/dist'] },
];
