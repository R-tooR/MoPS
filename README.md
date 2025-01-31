# MoPS
## Estimation of earthquake location using Geiger's method
Application simulates eartquake location finding process using Geiger's method. As a input it takes earthquake actual location, station locations and possibly estimated location.
### Example input
Input has to be JSON file, that contains
- `"data"` - array with data to simulation
- `"hypocenter"` - actual hypocenter location
- `"stations"` - stations location as array
- `"initial"` - optional, which location of earthquake is assumed as estimated at first iteration. If no provided, initial location is calculated as `[<mean_of_x_stations_coordinate>, <mean_of_y_stations_coordinate>, -1000]`
```
{
  "data": [
    {
      "hypocenter": [1300, 1550, -530],
      "stations": [[700, 1900, 0], [2500, 2000, 0], [1500, 2500, 0], [1850, 800, 0]],
      "initial":[1500, 2000, -700]
    },
    {
      "hypocenter": [3300, 5550, -1530],
      "stations": [[700, 1900, 0], [2500, 20, 0], [1800, 3300, 0], [1850, 800, 0]],
      "initial":[2000, 3000, -900]
    },
    {
      "hypocenter": [377, 550, -800],
      "stations": [[430, 190, 0], [250, 2000, 0], [1500, 2500, 0], [3850, 500, 0]],
      "initial":[1000, 800, -700]
    }
  ]
}
```

Run file with `python script.py <input_file> [output_file]`. If `[output_file]` is not provided, it saves output as `./output.txt`.
To make script work, at least 3 station should be provided.
### Example output
Output contains
- `"result"` - estimated location
- `"error"` - error rate
- `"history"` - results of all iterations
- `"history.result[0]"` - change of X coordinate
- `"history.result[1]"` - change of Y coordinate
- `"history.result[2]"` - change of Z coordinate
- `"history.error"` - change of error value
```
{
   "output":[
      {
         "result":[
            1306.9713940031968,
            1572.6925552812966,
            -516.7237747925539
         ],
         "error":4.009534254158231e-05,
         "history":{
            "result":[
               [
                  1302.7097448293623,
                  1305.395208949802,
                  1306.9643453631643,
                  1306.9712976507603,
                  1306.9713945922701,
                  1306.9713939990786,
                  1306.9713940031968
               ],
               [
                  1552.6166034763328,
                  1573.5408250943449,
                  1572.690748259741,
                  1572.6925267355734,
                  1572.6925553509325,
                  1572.6925552801433,
                  1572.6925552812966
               ],
               [
                  -659.1317384921931,
                  -526.9395953967945,
                  -516.7993143070441,
                  -516.7237394873056,
                  -516.7237752196451,
                  -516.7237747894258,
                  -516.7237747925539
               ]
            ],
            "error":[
               0.013923586155205642,
               0.022951058371422854,
               0.0014287580378523554,
               2.9112209558687652e-05,
               4.009267016357593e-05,
               4.00953233052892e-05,
               4.009534254158231e-05
            ]
         }
      },
      {
         "result":[
            3339.7552169399614,
            5602.606567166758,
            -1363.2058116608266
         ],
         "error":1.1706145163797865e-05,
         "history":{
            "result":[
               [
                  2526.1553789967193,
                  3529.4016019785686,
                  3346.028118077678,
                  3339.7598517199235,
                  3339.755209978202,
                  3339.755216944079,
                  3339.7552169399614
               ],
               [
                  4968.4952665798555,
                  5791.207399119048,
                  5625.583998063075,
                  5602.600826006188,
                  5602.606568455218,
                  5602.606567166533,
                  5602.606567166758
               ],
               [
                  -3719.099607745162,
                  -1981.676337557194,
                  -1371.0375753336948,
                  -1363.2329237628182,
                  -1363.2058166055408,
                  -1363.2058116566382,
                  -1363.2058116608266
               ]
            ],
            "error":[
               0.7444250366809861,
               0.16153772801644434,
               0.12776257788515522,
               0.006935321772633202,
               1.0239801098710366e-05,
               1.1706137372030634e-05,
               1.1706145163797865e-05
            ]
         }
      },
      {
         "result":[
            369.95362643809307,
            547.7346289298354,
            -779.4100114355475
         ],
         "error":0.00016027092144564603,
         "history":{
            "result":[
               [
                  424.7757355326519,
                  363.9396226082054,
                  369.98227935327685,
                  369.95365064613327,
                  369.9536270399865,
                  369.95362643863183,
                  369.95362643809307
               ],
               [
                  551.6398214779445,
                  537.8530221431779,
                  547.7591241932905,
                  547.7343900468974,
                  547.7346291670788,
                  547.7346289272942,
                  547.7346289298354
               ],
               [
                  -1031.7142911171868,
                  -788.5377267316821,
                  -779.5262315658686,
                  -779.4097898950284,
                  -779.410012436377,
                  -779.4100114337626,
                  -779.4100114355475
               ]
            ],
            "error":[
               0.0671180866537211,
               0.03195822016103059,
               0.002941449951916704,
               0.00017106060978518828,
               0.00016025960840805598,
               0.000160270975350027,
               0.00016027092144564603
            ]
         }
      }
   ]
}
```
