# SLAPStack-Controls
This package contains several control heuristics for the SLAPStack block 
stacking warehouse simulations. The code is hosted together with the simulation
on [github](https://github.com/malerinc/slapstack).

The following storage allocation problem (SLAP) policies are implemented:
* Closest Open Pure Lane (COPL)
* Class Based Popularity with the following stock keeping unit (SKU) popularity 
measures:
    * SKU Turnover time (indirectly proportional to popularity)
    * The historic number of picks per SKU (directly proportional to popularity)
    * The number of future SKU picks over the next planning period, e.g. week 
    (directly proportional to popularity)
    * The historic SKU throughput calculated as the sum of picks and deliveries 
    per SKU (directly proportional to popularity)
    * The future SKU throughput over the next planning period

The following unit load selection problem (ULSP) policies are implemented:
* Batch Last In First Out (BLIFO)


## Citing the Project
If you use `SLAPStack` or `WEPAStacks` in your research, you can cite this repository as follows:

```
@misc{rinciog2022slapstack
    author = {Rinciog, Alexandru and Pfrommer, Jakob and Morrissey Michael and Sohaib Zahid and Meyer Anne},
    title = {FabricatioRL},
    year = {2021},
    publisher = {GitHub},
    journal = {GitHub Repository},
    howpublished = {\url{https://github.com/malerinc/slapstack.git}},
}
```

## References
<a id="pfrommer2020">[1]</a> Pfrommer, J., Meyer, A.: Autonomously organized block stacking warehouses: A
review of decision problems and major challenges. Logistics Journal: Proceedings
2020(12) (2020)

<a id="rinciog2020">[2]</a> Rinciog, A., Meyer, A.: Fabricatio-rl: a reinforcement learning simulation frame-
work for production scheduling. In: 2021 Winter Simulation Conference (WSC).
pp. 1–12. IEEE (2021)



