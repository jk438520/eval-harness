# Harness eval shape

## Data structures

There are 2 data structures defined in evaluation harness. Both are descibed in json schemas.

## Harness parts

### Collator

Collator takes text as an input and outputs scores for all required categories. It expects text talks about Image A and Image B. If "Image A is sharper then Image B" then score for sharpness is 2.

### Statistics

Statistics part takes experiment (or experiments) and compares it against ground truth.

## Comments

 - In evaluation Motorolla should be Image A and Monalisa should be Image B.
 - This implies that general map of table to scores is :
  - <   ->  -2
  - <=  ->  -1
  - =   ->  0
  - >=  ->  1
  - >   ->  2