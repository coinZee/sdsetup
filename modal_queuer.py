import time
import threading
import queue

class QueueItem:
    def __init__(self, id, prompt, image_bytes, processed=False):
        self.id = id
        self.prompt = prompt
        self.image_bytes = image_bytes
        self.processed = processed

class Queue:
    def __init__(self):
        self.queue = queue.Queue()
        self.current_item_id = 0
        self.running = False
        self.lock = threading.Lock()

    def add_item(self, item):
        self.queue.put(item)
        if not self.running:
            self.running = True
            threading.Thread(target=self.process_queue, daemon=True).start()

    def process_queue(self):
        while not self.queue.empty() or self.running:
            time.sleep(3)
            if self.queue.empty():
                self.running = False
                break

            item = self.queue.get()
            if not item.processed:
                with self.lock:
                    self.current_item_id = int(item.id)
                item.processed = True
                print(f"Processing item {self.current_item_id} with prompt: {item.prompt}")
                time.sleep(5)
                print(f"Item {self.current_item_id} processed.")

    def get_current_item_id(self):
        with self.lock:
            return self.current_item_id

    def is_running(self):
        return self.running

    def check_position(self, id):
        with self.lock:
            position = self.queue.queue.index(next((x for x in self.queue.queue if x.id == id), None)) + 1
            if position == 0:
                print(f"Item with ID {id} not found in queue.")
            else:
                print(f"Item with ID {id} is at position {position} in the queue.")

def main():
    queue = Queue()

    while True:
        prompt = input("Enter a prompt (or type 'exit' to quit, or 'qt' followed by an ID to check position): ")
        if prompt == "exit":
            break

        # Check if the user wants to check their position in the queue
        if prompt.startswith("qt "):
            id = prompt[3:]
            queue.check_position(id)
            continue

        # Generate a unique ID and image bytes for the queue item
        id = str(int(time.time()))
        image_bytes = f"Image bytes for {prompt}"

        # Create a new queue item and add it to the queue
        item = QueueItem(id, prompt, image_bytes)
        queue.add_item(item)

        # Print a message indicating that the prompt is queued
        print(f"{prompt} is queued.")

if __name__ == "__main__":
    main()