// Copy lint — enforces docs/ui-copy.md: toasts and dialog titles are sentence case (no lowercase-first). Phase 9.

import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'fs';
import { resolve, dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const srcRoot = resolve(__dirname, '..');

interface Violation {
  file: string;
  offset: number;
  snippet: string;
  type: 'toast' | 'title';
}

function walkDir(dir: string, callback: (filePath: string) => void): void {
  const files = readdirSync(dir);
  for (const file of files) {
    const filePath = join(dir, file);
    const stat = statSync(filePath);
    if (stat.isDirectory()) {
      walkDir(filePath, callback);
    } else if (
      (file.endsWith('.ts') || file.endsWith('.tsx')) &&
      !file.endsWith('.test.ts') &&
      !file.endsWith('.test.tsx')
    ) {
      callback(filePath);
    }
  }
}

function lintCopy(): Violation[] {
  const violations: Violation[] = [];
  const toastRegex = /toast\.(?:success|error|info|warning|message)\(\s*'([a-z])/g;
  const titleRegex = /Dialog\.Title[^>]*>\s*([a-z])/g;

  walkDir(srcRoot, (filePath) => {
    const content = readFileSync(filePath, 'utf-8');
    const relPath = filePath.substring(srcRoot.length);

    // Check toasts
    toastRegex.lastIndex = 0;
    let match;
    while ((match = toastRegex.exec(content)) !== null) {
      const offset = match.index;
      const snippet = content.substring(offset, Math.min(offset + 60, content.length));
      violations.push({
        file: relPath,
        offset,
        snippet: snippet.replace(/\n/g, '\\n'),
        type: 'toast',
      });
    }

    // Check dialog titles
    titleRegex.lastIndex = 0;
    while ((match = titleRegex.exec(content)) !== null) {
      const offset = match.index;
      const snippet = content.substring(offset, Math.min(offset + 60, content.length));
      violations.push({
        file: relPath,
        offset,
        snippet: snippet.replace(/\n/g, '\\n'),
        type: 'title',
      });
    }
  });

  violations.sort((a, b) => {
    if (a.file !== b.file) return a.file.localeCompare(b.file);
    return a.offset - b.offset;
  });

  return violations;
}

describe('UI copy lint', () => {
  it('should not have lowercase-first toasts or dialog titles', () => {
    const violations = lintCopy();
    const message =
      violations.length > 0
        ? `Found ${violations.length} copy lint violations:\n${violations.map((v) => `  ${v.file}:${v.offset} [${v.type}] ${v.snippet}`).join('\n')}`
        : 'All copy lint checks passed';
    expect(violations, message).toEqual([]);
  });
});
