#include<bits/stdc++.h>
#include "random_tree.hpp"
#include "shortest_path.hpp"
#include "katz_distance.hpp"
#include "anchor_distance.hpp"
void main(int argc, char *argv[]){
	if(argv[2] == "random_tree"){
		random_tree(argv[0], argv[1], argc <= 3 ? 1000: atoi(argv[2]));
	}else if(argv[1] == "shortest_path"){
		shortest_path(argv[0], argv[1]);
	}else if(argv[1] == "katz_distance"){
		katz_distance(argv[0], argv[1], argv <= 3 ? 0.03 : atof(argv[2]), argv <= 4 ? 10 : atoi(argv[3]));
	}else if(argv[1] == "anchor_distance"){
		anchor_distance(argv[0], argv[1], argv <= 3 ? 10000 : atoi(argv[2]));
	}
}