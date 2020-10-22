#include<bits/stdc++.h>
#include "random_tree.hpp"
#include "shortest_path.hpp"
#include "katz_distance.hpp"
#include "anchor_distance.hpp"
int main(int argc, char *argv[]){
	srand(time(0));
	if(string(argv[1]) == "random_tree"){
		random_tree(argv[2], argv[3], argc < 5 ? 1000: atoi(argv[4]));
	}else if(string(argv[1]) == "shortest_path"){
		shortest_path(argv[2], argv[3]);
	}else if(string(argv[1]) == "katz_distance"){
		katz_distance(argv[2], argv[3], argc < 5 ? 0.03 : atof(argv[4]), argc < 6 ? 10 : atoi(argv[5]));
	}else if(string(argv[1]) == "anchor_distance"){
		anchor_distance(argv[2], argv[3], argc < 5 ? 10000 : atoi(argv[4]));
	}
	return 0;
}
