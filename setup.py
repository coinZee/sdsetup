import asyncio
import subprocess

async def main_exec():
    # Execute 'python -m modal setup' asynchronously
    setup_process = await asyncio.create_subprocess_shell('python -m modal setup')
    await setup_process.wait()  # Wait for the setup process to finish

    # Execute 'modal run flask_modal_server.py' asynchronously
    server_process = await asyncio.create_subprocess_shell('modal run flask_modal_server.py')
    await server_process.wait()  # Wait for the server process to finish

if __name__ == "__main__":
    asyncio.run(main_exec())
