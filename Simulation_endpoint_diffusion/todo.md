## choice of \[ \ceta \]
The core of the issue is ensuring that the "reach" of the contact inhibition signal is longer than the "reach" of the physical repulsion force.

Repulsion Range is a hard limit. In the simulation, it's defined by 2 * size. With size = 3.5, cells are forcefully kept at least 7.0 units apart.

Inhibition Range: This is a soft, decaying signal defined by the term \[ e^{−\ceta D} \]
 . The range of this

The rule of thumb is: You need the characteristic length of the inhibition to be comparable to, or larger than, the cell's diameter.

\( 
ceta
1 > 2*radius 
 \) kolla på gustavs modell. 


# todo 

- fit exponential to number of segmented cell n(t) and find residuals. look at the platemap for density. 
- comparison fogbank and other. 