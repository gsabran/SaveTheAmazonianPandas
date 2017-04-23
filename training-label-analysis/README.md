Simple analysis of label in the traning set

Mainly:
- 40480 training samples
- 17 different labels
- labels are:

| label | occurences |
|---|---|
| primary | 37840 |
| clear | 28203 |
| agriculture | 12338 |
| road | 8076 |
| water | 7262 |
| partly_cloudy | 7251 |
| cultivation | 4687 |
| habitation | 3662 |
| haze | 2695 |
| cloudy | 2330 |
| bare_ground | 859 |
| selective_logging | 340 |
| artisinal_mine | 339 |
| blooming | 332 |
| blow_down | 107 |
| slash_burn | 209 |
| conventional_mine | 100 |

- clear, cloudy, haze, partly_cloudy are exclusive
- one of the labels is one of (clear, cloudy, haze, partly_cloudy)
- no specific correlation noticed
- 442 different combinations
- max number of tag is 9
