package org.technodrome.diffuser.diffusion.helpers.loss;


public record Losses(WeightedL1 weightedL1,
                     WeightedL2 weightedL2,
                     ValueL1 valueL1,
                     ValueL2 valueL2) {

}