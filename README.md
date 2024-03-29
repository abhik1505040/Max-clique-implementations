# Max-clique implementations

## Available algorithms
* Ant colony optimization (aco) ([Fenet et al. 2012](https://www.researchgate.net/publication/2567745_Searching_for_Maximum_Cliques_with_Ant_Colony_Optimization))
* Branch and bound (bnb) ([Pattabiraman et al. 2003](https://arxiv.org/abs/1209.5818))

## Usage 
 
```bash
# to see available algorithms
python run.py -h
# to options available to a specific algorithm
python run.py aco -h
```
```bash
# run aco on all input graphs inside "input_graphs"
python run.py aco --input-dir input_graphs/
```

## Results

* ### Test graphs

Graph name|#Nodes|#Edges|Longest clique length
---|---|---|---|
anna|138|986|11
brock200_2|200|9876|12
brock200_4|200|13089|17
C125.9|125|6963|34
hamming8-4|256|20864|16
homer|561|3258|13
huck|74|602|11
keller4|171|9435|11
le450_15b|450|8169|15
le450_15c|450|16680|15
le450_15d|450|16750|15
le450_25a|450|8260|25
le450_25c|450|17343|25
le450_25d|450|17425|25
le450_5a|450|5714|5
le450_5b|450|5734|5
le450_5c|450|9803|5
le450_5d|450|9757|5
miles250|128|774|8
p_hat300-1|300|10933|8
p_hat300-2|300|21928|25

* ### Found clique sizes
    * ACO params
      * num ants = 7
      * taomin = 0.01
      * taomax = 4
      * alpha = 2
      * rho = 0.995
    * BNB params:
      * lb = 0  

*-> graph run didn't finish within time limit (20 mins)

Graph name (best)|Ant-clique -> avg(sdv)|Branch and bound
---|---|---
anna (11)|**11.00(0.00)**|**11**
brock200_2 (12)|11.33(0.47)|**12**
brock200_4 (17)|16.00(0.00)|**17**
C125.9 (34)|**34.00(0.00)**|*
hamming8-4 (16)|**16.00(0.00)**|**16**
homer (13)|**13.00(0.00)**|**13**
huck (11)|**11.00(0.00)**|**11**
keller4 (11)|**11.00(0.00)**|**11**
le450_15b (15)|**15.00(0.00)**|**15**
le450_15c (15)|**15.00(0.00)**|**15**
le450_15d (15)|**15.00(0.00)**|**15**
le450_25a (25)|**25.00(0.00)**|**25**
le450_25c (25)|**25.00(0.00)**|**25**
le450_25d (25)|**25.00(0.00)**|**25**
le450_5a (5)|**5.00(0.00)**|**5**
le450_5b (5)|**5.00(0.00)**|**5**
le450_5c (5)|**5.00(0.00)**|**5**
le450_5d (5)|**5.00(0.00)**|**5**
miles250 (8)|**8.00(0.00)**|**8**
p_hat300-1 (8)|**8.00(0.00)**|**8**
p_hat300-2 (25)|**25.00(0.00)**|*

* ### Running times (in ms)

Graph name (best)|Ant-clique -> nbCy(sdv)| Ant-clique -> time(sdv)|Branch and bound-> time
---|---|---|---|
anna (11)|7.67(2.05)|17.38(2.91)|**2.86**
brock200_2 (12)|380.67(428.10))|**5119.13**(5818.34)|10194.34
brock200_4 (17)|214.00(115.34)|**4587.07**(2452.60)|579099.92
C125.9 (34)|180.00(35.33)|**7312.11**(1569.49)|*
hamming8-4 (16)|99.66(40.54)|**2972.39**(1211.32)|898296.82
homer (13)|8.33(2.86)|35.60(9.52)|**2.69**
huck (11)|1.66(0.94)|6.33(2.99)|**0.51**
keller4 (11)|22.66(14.29)|**430.47**(268.46)|158878.92
le450_15b (15)|1.00(0.00)|**46.25**(5.40)|54.47
le450_15c (15)|1.00(0.00)|**88.35**(12.46)|556.35
le450_15d (15)|1.00(0.00)|**97.60**(13.06)|557.51
le450_25a (25)|1.00(0.00)|54.30(7.13)|**41.02**
le450_25c (25)|1.00(0.00)|**93.10**(4.86)|466.93
le450_25d (25)|1.00(0.00)|**92.79**(10.94)|494.23
le450_5a (5)|1.33(0.47)|**32.34**(3.23)|41.55
le450_5b (5)|1.00(0.00)|**29.13**(3.77)|41.11
le450_5c (5)|1.00(0.00)|**51.25**(9.27)|145.15
le450_5d (5)|1.00(0.00)|**48.91**(1.72)|140.18
miles250 (8)|4.66(2.49)|10.19(3.34)|**0.78**
p_hat300-1 (8)|81.00(38.28)|**1119.92**(493.61)|1408.27
p_hat300-2 (25)|162.00(26.19)|**5640.97**(916.14)|*
