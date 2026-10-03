import asyncio

# Tracks running consumer asyncio tasks: {name: task}
_consumer_tasks: dict[str, asyncio.Task] = {}


def get_consumer_tasks() -> dict[str, asyncio.Task]:
    return _consumer_tasks
