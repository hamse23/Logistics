# Washing Machine Production Optimization

This project determines the optimal production plan for three washing machine types — Type A, Type B, and Type C — with the goal of maximizing overall revenue while considering manufacturing costs, production constraints, and raw material availability.
### Problem Summary
Each washing machine type has:
- Different profit per unit
- Different manufacturing costs
- Different iron consumption
#### Additional constraints:
- Maximum iron available: 4800 kg
- The number of Type A machines must be at least twice Type B
- The number of Type A machines must not exceed Type C
- Iron cost: €4/kg
#### The objective is to maximize:
(Sales profit) − (Manufacturing cost + Iron cost)
The problem is formulated as a linear programming model and solved using AMPL + Gurobi.
### Results
1) Base Scenario (No additional iron purchased)
- Type A: 1372 units
- Type B: 684 units
- Type C: 1372 units
- Maximum revenue: €51,428
2) With €800 additional budget (extra iron purchased)
Extra iron purchased: about 114.29 kg
- Type A: 1404 units
- Type B: 702 units
- Type C: 1404 units
- Maximum revenue: €52,650
⇒ Extra revenue gain: €1,222
This shows that even a small investment in additional raw materials can increase production capacity and overall revenue.
## Tools Used
- AMPL modeling language
- Gurobi solver
- PDF report summarizing model and results
## Repository Contents
- AMPL model file
- Data file
- Run/execution script
- Solver output log
- Final report (PDF)
