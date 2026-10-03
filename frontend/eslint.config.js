import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import reactHooks from 'eslint-plugin-react-hooks';

export default [
  {
    ignores: [
      'node_modules/**',
      'dist/**',
      'src/components/common/uiPrimitives.tsx',
      'src/lib/statusColors.ts',
      'tailwind.config.ts',
    ],
  },
  {
    files: ['src/**/*.{ts,tsx}'],
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
        ecmaFeatures: {
          jsx: true,
        },
      },
    },
    plugins: {
      '@typescript-eslint': tseslint.plugin,
      'react-hooks': reactHooks,
    },
    rules: {
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
      'no-restricted-syntax': [
        'error',
        {
          selector: 'Property[key.name=/^(color|backgroundColor|borderColor|background|fill|stroke)$/] > Literal[value=/^#[0-9A-Fa-f]{3,8}$/]',
          message: 'Raw hex in style prop. Use a UI_COLORS / STATUS_COLORS token instead.',
        },
        {
          selector: 'Property[key.name=/^(border|borderTop|borderRight|borderBottom|borderLeft|outline|boxShadow)$/] > Literal[value=/.*#[0-9A-Fa-f]{3,8}.*/]',
          message: 'Raw hex in border/shadow shorthand. Use a token template literal: `1px solid ${UI_COLORS.border}`.',
        },
        {
          selector: 'JSXAttribute[name.name="className"] Literal[value=/.*#[0-9A-Fa-f]{3,8}.*/]',
          message: 'Raw hex in className. Use a Tailwind huddleroom-* token class instead.',
        },
      ],
    },
  },
];
