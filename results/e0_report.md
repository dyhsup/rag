# E0-core: pointwise top-k is redundancy-blind (oracle grounding labels)

- questions: 10　|　pool = 2 gold + their equivalent rewrites + 6 negatives

| k | pointwise coverage | greedy coverage | gap |
|---|---|---|---|
| 1 | 0.250 | 0.500 | +0.250 |
| 2 | 0.500 | 1.000 | +0.500 |
| 3 | 0.550 | 1.000 | +0.450 |
| 4 | 0.650 | 1.000 | +0.350 |
| 5 | 0.750 | 1.000 | +0.250 |
| 6 | 0.750 | 1.000 | +0.250 |

At k=2: pointwise 0.500 vs coverage 1.000

## per-question at k=2

| question | pointwise picks | coverage picks |
|---|---|---|
| What nationality was Oliver Reed's character in the film Roy | ['NEG', 'alias'] | ['alias', 'relation'] |
| Pacific Mozart Ensemble performed which German composer's De | ['Kurt Weill', 'relation'] | ['Kurt Weill', 'inversion'] |
| Who released the song "With or Without You" first, Jai McDow | ['inversion', 'Believe (Jai McDowall album)'] | ['inversion', 'Believe (Jai McDowall album)'] |
| What Kentucky county has a population of 60,316 and features | ['Lake Louisvilla, Louisville', 'dispersal'] | ['Lake Louisvilla, Louisville', 'Oldham County, Kentucky'] |
| Para Hills West, South Australia lies within a city with wha | ['NEG', 'inversion'] | ['inversion', 'City of Salisbury'] |
| In what year was the narrator of "Blackadder's Christmas Car | ['NEG', 'relation'] | ['relation', 'Hugh Laurie'] |
| Thomas Fitch defended John Henry Holliday when he was accuse | ['Thomas Fitch (politician)', 'dispersal'] | ['Thomas Fitch (politician)', 'dispersal'] |
| The actor that played Gutthi on Comedy nights with Kapil Sho | ['NEG', 'inversion'] | ['inversion', 'inversion'] |
| Victor Varnada directed Twisted Fortune, which starred which | ['NEG', 'NEG'] | ['Twisted Fortune', 'Charlie Murphy'] |
| After his curacy at the village that is a suburb of Scunthor | ['Frodingham, Lincolnshire', 'relation'] | ['Frodingham, Lincolnshire', 'relation'] |
