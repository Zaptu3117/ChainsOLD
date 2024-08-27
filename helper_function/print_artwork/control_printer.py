import asyncio
from web3 import Web3
from web3.middleware import geth_poa_middleware
import json
from PIL import Image, ImageWin
import win32print
import win32ui

# Ethereum node URL (replace with your own)
ETH_NODE_URL = "https://mainnet.infura.io/v3/YOUR-PROJECT-ID"

# Contract address (replace with your deployed contract address)
CONTRACT_ADDRESS = "0x1234567890123456789012345678901234567890"

# ABI of the contract (replace with your contract's ABI)
CONTRACT_ABI = json.loads('''
[
    {"inputs":[{"internalType":"uint256","name":"_initialCost","type":"uint256"}],"stateMutability":"nonpayable","type":"constructor"},
    {"anonymous":false,"inputs":[{"indexed":false,"internalType":"uint256","name":"newCost","type":"uint256"}],"name":"CostUpdated","type":"event"},
    {"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"requester","type":"address"},{"indexed":false,"internalType":"string","name":"message","type":"string"},{"indexed":false,"internalType":"uint256","name":"timestamp","type":"uint256"}],"name":"PrintRequested","type":"event"},
    {"inputs":[],"name":"getPrintJobCount","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"uint256","name":"index","type":"uint256"}],"name":"getPrintJob","outputs":[{"internalType":"address","name":"","type":"address"},{"internalType":"string","name":"","type":"string"},{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"owner","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"printCost","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"string","name":"message","type":"string"}],"name":"requestPrint","outputs":[],"stateMutability":"payable","type":"function"},
    {"inputs":[],"name":"totalPrints","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"uint256","name":"newCost","type":"uint256"}],"name":"updatePrintCost","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[],"name":"withdrawFunds","outputs":[],"stateMutability":"nonpayable","type":"function"}
]
''')

# Initialize Web3
w3 = Web3(Web3.HTTPProvider(ETH_NODE_URL))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)  # Needed for some networks like Binance Smart Chain

# Create contract instance
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)

def print_image(printer_name, image_path, width, height):
    """Function to print an image."""
    hprinter = win32print.OpenPrinter(printer_name)
    try:
        hdc = win32ui.CreateDC()
        hdc.CreatePrinterDC(printer_name)
        hdc.StartDoc(image_path)
        hdc.StartPage()
        image = Image.open(image_path)
        dib = ImageWin.Dib(image)
        dib.draw(hdc.GetHandleOutput(), (0, 0, width, height))
        hdc.EndPage()
        hdc.EndDoc()
        print(f"Image printed: {image_path}")
    except Exception as e:
        print(f"Error printing image: {e}")
    finally:
        win32print.ClosePrinter(hprinter)

async def log_loop(event_filter, poll_interval):
    """Loop to continuously check for new events."""
    while True:
        for event in event_filter.get_new_entries():
            handle_event(event)
        await asyncio.sleep(poll_interval)

def handle_event(event):
    """Handle the PrintRequested event."""
    print("Print Requested Event:")
    print(f"Requester: {event['args']['requester']}")
    print(f"Message: {event['args']['message']}")
    print(f"Timestamp: {event['args']['timestamp']}")
    
    # Trigger the print job
    print_image("Epson Expression Photo XP-15000", "path/to/your/image.jpg", 3508, 4961)

async def main():
    """Main function to set up the event filter and start the loop."""
    print("Starting to listen for PrintRequested events...")
    event_filter = contract.events.PrintRequested.create_filter(fromBlock='latest')
    loop = asyncio.get_event_loop()
    try:
        await log_loop(event_filter, 2)
    finally:
        print("Stopping event listener...")

if __name__ == "__main__":
    asyncio.run(main())