#include<bits/stdc++.h>
#include "dsu.hpp"
#include "tree.hpp"
using namespace std;
struct P{
	int x, y;
	double z;
};
void random_tree(char *in_dir, char *out_dir, int num){
	int num_nodes, train_pos_m, train_neg_m, valid_pos_m, valid_neg_m, test_pos_m, test_neg_m;
	Tree tree;
	freopen(in_dir, "r", stdin);
	scanf("%d", &num_nodes);
	int shuffle[num_nodes + 10];
	scanf("%d", &train_pos_m);
	P train_pos[train_pos_m + 10], train_pos2[train_pos_m + 10];
	for(int i = 0; i < train_pos_m; i++){
		scanf("%d%d", &train_pos[i].x, &train_pos[i].y);
		train_pos2[i] = train_pos[i];
	}
	scanf("%d", &train_neg_m);
	P train_neg[train_neg_m + 10];
	for(int i = 0; i < train_neg_m; i++)
		scanf("%d%d", &train_neg[i].x, &train_neg[i].y);
	scanf("%d", &valid_pos_m);
	P valid_pos[valid_pos_m + 10];
	for(int i = 0; i < valid_pos_m; i++)
		scanf("%d%d", &valid_pos[i].x, &valid_pos[i].y);
	scanf("%d", &valid_neg_m);
	P valid_neg[valid_neg_m + 10];
	for(int i = 0; i < valid_neg_m; i++)
		scanf("%d%d", &valid_neg[i].x, &valid_neg[i].y);
	scanf("%d",&test_pos_m);
	P test_pos[test_pos_m + 10];
	for(int i = 0; i < test_pos_m; i++)
		scanf("%d%d", &test_pos[i].x, &test_pos[i].y);
	scanf("%d",&test_neg_m);
	P test_neg[test_neg_m + 10];
	for(int i = 0; i < test_neg_m; i++)
		scanf("%d%d", &test_neg[i].x, &test_neg[i].y);
	fclose(stdin);
	for(int sample = 0; sample < num; sample++){
		random_shuffle(train_pos2, train_pos2 + train_pos_m);
		DSU dsu(num_nodes);
		for(int i = 0; i < train_pos_m; i++)
			if(dsu.merge(train_pos2[i].x, train_pos2[i].y))
				tree.add_edge(train_pos2[i].x, train_pos2[i].y);
		for(int i = 0; i < num_nodes; i++)
			shuffle[i] = i;
		random_shuffle(shuffle, shuffle + num_nodes);
		num_nodes++;
		tree.init(num_nodes);
		for(int i = 0; i < num_nodes - 1; i++){
			int x = union_find(shuffle[i]);
			int y = union_find(num_nodes - 1);
			if(x != y){
				father[x] = y;
				tree.add_edge(shuffle[i], num_nodes - 1);
			}
		}
		tree.process();
		for(int i = 0; i < train_pos_m; i++)
			train_pos[i].z += tree.dis(train_pos[i].x, train_pos[i].y);
		for(int i = 0; i < train_neg_m; i++)
			train_neg[i].z += tree.dis(train_neg[i].x, train_neg[i].y);
		for(int i = 0; i < valid_pos_m; i++)
			valid_pos[i].z += tree.dis(valid_pos[i].x, valid_pos[i].y);
		for(int i = 0; i < valid_neg_m; i++)
			valid_neg[i].z += tree.dis(valid_neg[i].x, valid_neg[i].y);
		for(int i = 0; i < test_pos_m; i++)
			test_pos[i].z += tree.dis(test_pos[i].x, test_pos[i].y);
		for(int i = 0; i < test_neg_m; i++)
			test_neg[i].z += tree.dis(test_neg[i].x, test_neg[i].y);
	}
	freopen(out_dir + "train_pos_avg.txt", "w", stdout);
	for(int i = 0; i < train_pos_m; i++)
		printf("%.15lf\n", train_pos[i].z / num);
	fclose(stdout);
	freopen(out_dir + "train_neg_avg.txt", "w", stdout);
	for(int i = 0; i < train_neg_m; i++)
		printf("%.15lf\n", train_neg[i].z / num);
	fclose(stdout);
	freopen(out_dir + "valid_pos_avg.txt", "w", stdout);
	for(int i = 0; i < valid_pos_m; i++)
		printf("%.15lf\n", valid_pos[i].z / num);
	fclose(stdout);
	freopen(out_dir + "valid_neg_avg.txt", "w", stdout);
	for(int i = 0; i < valid_neg_m; i++)
		printf("%.15lf\n", valid_neg[i].z / num);
	fclose(stdout);
	freopen(out_dir + "test_pos_avg.txt", "w", stdout);
	for(int i = 0; i < test_pos_m; i++)
		printf("%.15lf\n", test_pos[i].z / num);
	fclose(stdout);
	freopen(out_dir + "test_neg_avg.txt", "w", stdout);
	for(int i = 0; i < test_neg_m; i++)
		printf("%.15lf\n", test_neg[i].z / num);
	fclose(stdout);
}