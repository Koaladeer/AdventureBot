# Playing Zork1

This project aims to develop an intelligent agent capable of playing the classic text-based adventure game Zork1, leveraging the [Jericho](https://jericho-py.readthedocs.io/en/latest/tutorial_quick.html) library. Jericho provides a robust interface for interacting with Z-machine games, allowing for programmatic control and observation of game states.

**Jericho Project Links:**

* [Jericho Quick Tutorial](https://jericho-py.readthedocs.io/en/latest/tutorial_quick.html)

* [JerichoWorld GitHub Repository](https://github.com/JerichoWorld/JerichoWorld)

## Description

This project focuses on building and training an AI agent to navigate and solve Zork1. It involves data collection from both walkthroughs and random exploration, agent development for game interaction, and analysis of the collected data to understand game dynamics and agent performance.

## Requirements & Setup

To run this project, you will need the following:

* **Operating System:** Linux

* **Python:** Version 3.9 or higher

Follow these steps to set up your environment:

1. **Install Jericho:**

   ```bash
   pip3 install jericho
   ```

2.  **Download SpaCy Model:**

    ```bash
    python3 -m spacy download en_core_web_sm
    ```

3.  **Obtain Z-Machine Game ROMs (specifically Zork1):**

    ```bash
    wget [https://github.com/BYU-PCCL/z-machine-games/archive/master.zip](https://github.com/BYU-PCCL/z-machine-games/archive/master.zip)
    unzip master.zip
    ```

    Place the `zork1.z5` file (found within the unzipped `z-machine-games-master/jericho-game-suite/` directory) in the appropriate path as referenced by your scripts (e.g., `../games/z-machine-games-master/jericho-game-suite/zork1.z5`).

4.  **JerichoWorld Framework for Training Data:**
    To build the training datasets (e.g., walkthrough and random exploration data), this project utilizes functionalities provided by the JerichoWorld framework. Ensure you have access to or have set up the necessary components of JerichoWorld if you intend to generate new training data.

## Repository Structure

This repository is organized to separate different aspects of the project:

| File/Directory Prefix | Description |
| :-------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gemini_...`          | Files prefixed with `gemini_` are generated or heavily influenced by GenAI (e.g., Gemini) to facilitate brainstorming, rapid prototyping, and faster iteration cycles. |
| `Agent_...`           | These files contain the core game loop logic and the implementation of the AI agent, including its decision-making processes and interactions with the game.         |
| `Trainer_...`         | Scripts within this category are responsible for training the agent. This includes defining training environments, reward functions, and learning algorithms.          |
| `Analysis_...`        | This section is dedicated to data exploration, statistical analysis, and plotting of insights derived from the game interaction data and agent performance.            |

