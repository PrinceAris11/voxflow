import asyncio
import json
import websockets

async def main():
    async with websockets.connect('ws://127.0.0.1:8000/ws') as ws:
        await ws.send(json.dumps({'type': 'ping'}))
        print('sent ping')
        reply = await ws.recv()
        print('server:', reply)
        await ws.send(json.dumps({'type': 'audio', 'data': 'mock-audio-bytes'}))
        reply2 = await ws.recv()
        print('server:', reply2)
        await ws.send(json.dumps({'type': 'transcript', 'text': 'Um basically we are gonna launch the product you know'}))
        reply3 = await ws.recv()
        print('server transcript:', reply3)

if __name__ == '__main__':
    asyncio.run(main())
