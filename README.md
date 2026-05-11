# MACMAT: Memory-augmented Asynchronous Class-based Multi-Agent Transformer

**MACMAT** is a fully mapless, decentralized Multi-Agent Reinforcement Learning (MARL) framework designed for the robust coordination of heterogeneous robotic teams in highly dynamic environments. It addresses the challenges of asynchronous decision-making, partial observability, and intermittent communication without the need for a global map.

By utilizing continuous-time liquid memory networks to process asynchronous high-level waypoints, MACMAT enables resilient and scalable multi-robot coordination, particularly in scenarios where maintaining a shared occupancy grid map is infeasible.

The framework is validated in **MiniSAR**, a custom 2D grid-world simulation environment for multi-robot Search and Rescue (SAR).

---

## 🏗️ Framework Architecture

MACMAT operates on a **Centralized Training with Decentralized Execution (CTDE)** paradigm. Each agent operates using a modular onboard architecture consisting of three core components:

1.  **Spatial Awareness Module:** Processes local sensory data, relying on local observation streams and Visual-Inertial Odometry (VIO) estimates to maintain state awareness without a global map.
2.  **High-Level Decision-Making Module (MACMAT):** The strategic "brain" that selects asynchronous macro-actions (e.g., "Navigate to relative Waypoint X").
3.  **Low-Level Motion Control Module:** A local planner that executes the macro-action by generating primitive motor commands to reach the selected waypoint.

### The MACMAT Core

MACMAT replaces explicit map-building with a **Recurrent Neural Network (RNN)** architecture integrated with a Transformer backbone.

* **Input:** Multi-channel local semantic grid, agent pose, and relative target positions.
* **Memory Mechanisms:** The framework supports and evaluates several RNN variants:
    * **LSTM:** A standard discrete-time choice.
    * **Closed-form Continuous-time (CfC):** A continuous-time neural network layer (dense).
    * **Neural Circuit Policies (NCP):** A CfC with biologically inspired sparse wiring.
    * **Mixed-Memory (MixedNCP):** A novel architecture combining an **LSTM** with a sparse **CfC**. This allows agents to capture temporal dependencies over highly irregular, asynchronous intervals while benefiting from the long-term gating of LSTMs.
* **Heterogeneity:** The model uses **Class-Specific Action Heads** to learn distinct, specialized behaviors for different robot types (e.g., Rescuers vs. Scouts) without requiring explicit class identifiers in the observation input.

---

## 🎮 MiniSAR Environment

The framework is tested in **MiniSAR**, a custom search-and-rescue testbed modified from the [Minigrid](https://minigrid.farama.org/index.html) environment.

* **Features:** Procedurally generated cluttered rooms, dynamic targets, varying obstacle types, and strict "fog of war" local visibility.
* **Tasks:** Supports dynamic multi-target search and rescue scenarios requiring tight coordination between heterogeneous teams (e.g., fast Scouts for exploration and specialized Rescuers for target retrieval).

### MiniSAR demos

Below, MACMAT-driven heterogeneous teams coordinate in MiniSAR at three grid scales (larger worlds stress exploration and long-horizon coordination).

<p align="center">
  <img src="hetmarl/docs/macmat_15x15.gif" alt="MACMAT in MiniSAR on a 15 by 15 grid" width="30%" />
  &nbsp;
  <img src="hetmarl/docs/macmat_20x20.gif" alt="MACMAT in MiniSAR on a 20 by 20 grid" width="30%" />
  &nbsp;
  <img src="hetmarl/docs/macmat_28x28.gif" alt="MACMAT in MiniSAR on a 28 by 28 grid" width="30%" />
</p>

<p align="center">
  <sub><strong>15×15</strong> · <strong>20×20</strong> · <strong>28×28</strong></sub>
</p>

---

## 🚀 Installation

The framework requires Python 3.9+.

```bash
conda create -n macmat python=3.9
conda activate macmat
pip3 install -r requirements.txt
