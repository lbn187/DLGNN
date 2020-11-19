#include "random_tree.hpp"
struct P{
	int x, y;
	double z;
};
void random_tree(char *in_dir, char *out_dir, int num, bool use_val){
	int num_nodes, train_pos_m, train_neg_m, valid_pos_m, valid_neg_m, test_pos_m, test_neg_m;
	vector<P> train_pos, train_pos2, train_pos3, train_neg, valid_pos, valid_neg, test_pos, test_neg;
	freopen(in_dir, "r", stdin);
	scanf("%d", &num_nodes);
	scanf("%d", &train_pos_m);
	train_pos.resize(train_pos_m);
	train_pos2.resize(train_pos_m);
	for(int i = 0; i < train_pos_m; i++){
		scanf("%d%d", &train_pos[i].x, &train_pos[i].y);
		train_pos2[i] = train_pos[i];
		train_pos3.push_back(train_pos[i]);
	}
	scanf("%d", &train_neg_m);
	train_neg.resize(train_neg_m);
	for(int i = 0; i < train_neg_m; i++)
		scanf("%d%d", &train_neg[i].x, &train_neg[i].y);
	scanf("%d", &valid_pos_m);
	valid_pos.resize(valid_pos_m);
	for(int i = 0; i < valid_pos_m; i++){
		scanf("%d%d", &valid_pos[i].x, &valid_pos[i].y);
		train_pos3.push_back(valid_pos[i]);
	}
	scanf("%d", &valid_neg_m);
	valid_neg.resize(valid_neg_m);
	for(int i = 0; i < valid_neg_m; i++)
		scanf("%d%d", &valid_neg[i].x, &valid_neg[i].y);
	scanf("%d",&test_pos_m);
	test_pos.resize(test_pos_m);
	for(int i = 0; i < test_pos_m; i++)
		scanf("%d%d", &test_pos[i].x, &test_pos[i].y);
	scanf("%d",&test_neg_m);
	test_neg.resize(test_neg_m);
	for(int i = 0; i < test_neg_m; i++)
		scanf("%d%d", &test_neg[i].x, &test_neg[i].y);
	fclose(stdin);
	num_nodes++;
	for(int sample = 0; sample < num; sample++){
		random_shuffle(train_pos2.begin(), train_pos2.end());
		DSU dsu(num_nodes);
		Tree tree(num_nodes);
		for(int i = 0; i < train_pos_m; i++)
			if(dsu.merge(train_pos2[i].x, train_pos2[i].y))
				tree.add_edge(train_pos2[i].x, train_pos2[i].y);
		for(int i = 0; i < num_nodes - 1; i++)
			if(dsu.merge(i, num_nodes - 1))
				tree.add_edge(i, num_nodes - 1);
		for(int i = 0; i < train_pos_m; i++)
			train_pos[i].z += tree.dis(train_pos[i].x, train_pos[i].y);
		for(int i = 0; i < train_neg_m; i++)
			train_neg[i].z += tree.dis(train_neg[i].x, train_neg[i].y);
		for(int i = 0; i < valid_pos_m; i++)
			valid_pos[i].z += tree.dis(valid_pos[i].x, valid_pos[i].y);
		for(int i = 0; i < valid_neg_m; i++)
			valid_neg[i].z += tree.dis(valid_neg[i].x, valid_neg[i].y);
		if(use_val){
			random_shuffle(train_pos3.begin(), train_pos3.end());
			DSU new_dsu(num_nodes);
			Tree new_tree(num_nodes);
			for(P edge : train_pos3)
				if(new_dsu.merge(edge.x, edge.y))
					new_tree.add_edge(edge.x, edge.y);
			for(int i = 0; i < num_nodes - 1; i++)
				if(new_dsu.merge(i, num_nodes - 1))
					new_tree.add_edge(i, num_nodes - 1);
			for(int i = 0; i < test_pos_m; i++)
				test_pos[i].z += new_tree.dis(test_pos[i].x, test_pos[i].y);
			for(int i = 0; i < test_neg_m; i++)
				test_neg[i].z += new_tree.dis(test_neg[i].x, test_neg[i].y);	
		}else{
			for(int i = 0; i < test_pos_m; i++)
				test_pos[i].z += tree.dis(test_pos[i].x, test_pos[i].y);
			for(int i = 0; i < test_neg_m; i++)
				test_neg[i].z += tree.dis(test_neg[i].x, test_neg[i].y);
		}
	}
	freopen((string(out_dir) + string("random_tree_train_pos.txt")).c_str(), "w", stdout);
	for(int i = 0; i < train_pos_m; i++)
		printf("%.15lf\n", train_pos[i].z / num);
	fclose(stdout);
	freopen((string(out_dir) + string("random_tree_train_neg.txt")).c_str(), "w", stdout);
	for(int i = 0; i < train_neg_m; i++)
		printf("%.15lf\n", train_neg[i].z / num);
	fclose(stdout);
	freopen((string(out_dir) + string("random_tree_valid_pos.txt")).c_str(), "w", stdout);
	for(int i = 0; i < valid_pos_m; i++)
		printf("%.15lf\n", valid_pos[i].z / num);
	fclose(stdout);
	freopen((string(out_dir) + string("random_tree_valid_neg.txt")).c_str(), "w", stdout);
	for(int i = 0; i < valid_neg_m; i++)
		printf("%.15lf\n", valid_neg[i].z / num);
	fclose(stdout);
	freopen((string(out_dir) + string("random_tree_test_pos.txt")).c_str(), "w", stdout);
	for(int i = 0; i < test_pos_m; i++)
		printf("%.15lf\n", test_pos[i].z / num);
	fclose(stdout);
	freopen((string(out_dir) + string("random_tree_test_neg.txt")).c_str(), "w", stdout);
	for(int i = 0; i < test_neg_m; i++)
		printf("%.15lf\n", test_neg[i].z / num);
	fclose(stdout);
}
