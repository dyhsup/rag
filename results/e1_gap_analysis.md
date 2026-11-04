# First-stage failure analysis (MuSiQue, 200 questions)

- hops: 540

## gold rank under dense (MiniLM)

| rank bucket | hops | Jaccard(question, gold) | Jaccard(hop, gold) |
|---|---|---|---|
| 1 | 140 | 0.123 | 0.094 |
| 2-5 | 104 | 0.100 | 0.070 |
| 6-50 | 151 | 0.086 | 0.053 |
| 51-100 | 47 | 0.078 | 0.050 |
| >100 | 98 | 0.074 | 0.038 |

## gold rank under BM25

| rank bucket | hops | Jaccard(question, gold) | Jaccard(hop, gold) |
|---|---|---|---|
| 1 | 101 | 0.137 | 0.103 |
| 2-5 | 67 | 0.108 | 0.077 |
| 6-50 | 142 | 0.095 | 0.071 |
| 51-100 | 34 | 0.090 | 0.051 |
| >100 | 196 | 0.070 | 0.036 |

## examples of dense misses (rank > 100): 98

- Who is the spouse of the actor of Ethan in A Dog's Purpose? | gold=Flesh and Bone (film) | rank=119 | jac_q=0.12
- Who is the spouse of the author of Queen of the Elephants? | gold=Clio Goldsmith | rank=105 | jac_q=0.08
- How many students attend the university that Edith Humphrey was educated? | gold=Switzerland | rank=157 | jac_q=0.05
- Who was the child of the Italian navigator who sailed for England, and explored  | gold=Paula Santiago | rank=373 | jac_q=0.07
- What record label is responsible for the performer of "Sanctuary"? | gold=Charlie Musselwhite | rank=209 | jac_q=0.04
- Who is the spouse of the actress who played the fairy god mother in Shrek 2? | gold=Mirrorball (TV pilot) | rank=373 | jac_q=0.06
- How were the people that the Somali Muslim Ajuran Empire made coins to proclaim  | gold=Myanmar | rank=160 | jac_q=0.02
- Where is the headquarters of the Radio Television of the country that has a co-o | gold=Media in Pristina | rank=116 | jac_q=0.10
