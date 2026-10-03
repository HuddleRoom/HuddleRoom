#!/bin/bash
# Simple echo agent for CLI adapter testing
cat "$1" || exit 1
echo "Task completed successfully"
exit 0
