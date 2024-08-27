// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract PrinterTrigger {
    address public owner;
    uint256 public printCost;
    uint256 public totalPrints;

    struct PrintJob {
        address requester;
        string message;
        uint256 timestamp;
    }

    PrintJob[] public printJobs;

    event PrintRequested(address indexed requester, string message, uint256 timestamp);
    event CostUpdated(uint256 newCost);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }

    constructor(uint256 _initialCost) {
        owner = msg.sender;
        printCost = _initialCost;
    }

    function requestPrint(string memory message) public payable {
        require(msg.value >= printCost, "Insufficient payment");
        
        uint256 timestamp = block.timestamp;
        printJobs.push(PrintJob(msg.sender, message, timestamp));
        totalPrints++;

        emit PrintRequested(msg.sender, message, timestamp);

        if (msg.value > printCost) {
            payable(msg.sender).transfer(msg.value - printCost);
        }
    }

    function updatePrintCost(uint256 newCost) public onlyOwner {
        printCost = newCost;
        emit CostUpdated(newCost);
    }

    function withdrawFunds() public onlyOwner {
        payable(owner).transfer(address(this).balance);
    }

    function getPrintJobCount() public view returns (uint256) {
        return printJobs.length;
    }

    function getPrintJob(uint256 index) public view returns (address, string memory, uint256) {
        require(index < printJobs.length, "Invalid index");
        PrintJob memory job = printJobs[index];
        return (job.requester, job.message, job.timestamp);
    }
}