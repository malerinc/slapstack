# SLAPStack-Controls
This package contains several control heuristics associated with the SLAPStack 
block stacking warehouse (BSW) simulation. The code is hosted together with the 
simulation on [github](https://github.com/malerinc/slapstack).
SLAPStack contains two use cases, namely `WEPAStacks` and `Crossstacks`. 
See the linked repository for more information.

For `WEPAStacks`, the following storage location allocation problem (SLAP) 
strategies were implemented and tested (a comparison of these strategies is 
available through [[3]](#pfrommer2022)):
* Closest open pure lane (COPL) and
* Class-based popularity with the following stock keeping unit (SKU) popularity 
measures:
    * SKU turnover time (indirectly proportional to popularity)
    * The historic number of picks per SKU (directly proportional to popularity)
    * The number of future SKU picks over the next planning period, e.g. week 
    (directly proportional to popularity)
    * The historic SKU throughput calculated as the sum of picks and deliveries 
    per SKU (directly proportional to popularity)
    * The future SKU throughput over the next planning period

For `CrossStacks` (publication pending), the implemented strategies are:
* Closest to destination (CTD),
* Closest open location (COL),
* Random location (RND), and
* Two dual command cycle inspired heuristics:
  * Closest to the next delivery order (CTNR)
  * Shortest leg (SL)

Note that the `CrossStacks` SLAP strategies could be applied to the `WEPAStacks` 
use case and vice-versa, however this application has not yet been tested.

The following unit load selection problem (ULSP) policies are implemented:
* Batch Last In First Out (BLIFO)


## Citing the Project
If you use `SLAPStack`, `WEPAStacks` or `CrossStacks` in your research, you can 
cite this repository as follows:

```
@misc{rinciog2023slapstack
    author = {Rinciog, Alexandru and Pfrommer, Jakob and Morrissey Michael 
      and Sohaib Zahid and Vasileva, Anna and Ogorelysheva, Natalia and 
      Rathod, Hardik and Meyer Anne},
    title = {SLAPStack},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub Repository},
    howpublished = {\url{https://github.com/malerinc/slapstack.git}},
}
```

## References
<a id="pfrommer2020">[1]</a> Pfrommer, J., Meyer, A.: Autonomously organized 
block stacking warehouses: A review of decision problems and major challenges. 
Logistics Journal: Proceedings 2020(12) (2020)

<a id="rinciog2020">[2]</a> Rinciog, A., Meyer, A.: Fabricatio-rl: 
A reinforcement learning simulation framework for production scheduling. 
In: 2021 Winter Simulation Conference (WSC).
pp. 1–12. IEEE (2021)

<a id="pfrommer2022">[3]</a> Pfrommer, J.; Rinciog, A.; Zahid, S.; Morrissey, M; 
Meyer A. (2022): SLAPStack: A Simulation Framework and a Large-Scale Benchmark 
Use Case for Autonomous Block Stacking Warehouses. 
International Conference on Computational Logistics (ICCL) 2022.



