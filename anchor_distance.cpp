#include"anchor_distance.hpp"
#include"dsu.hpp"
struct P{
	int x, y, z;
};
int R(int maxv){
	int x = rand() % 32768;
	int y = rand() % 32768;
	return (x * 32768 + y) % maxv;
}
void anchor_distance(char *in_dir, char *out_dir, int num){
	int num_nodes, train_pos_m, train_neg_m, valid_pos_m, valid_neg_m, test_pos_m, test_neg_m;
	vector<P> train_pos, train_pos2, train_neg, valid_pos, valid_neg, test_pos, test_neg;
	vector<vector<int> >edge;
	vector<int> vis, d, vertices1, vertices2;
	freopen(in_dir, "r", stdin);
	scanf("%d", &num_nodes);
	edge.resize(num_nodes);
	vis.resize(num_nodes);
	d.resize(num_nodes);
	scanf("%d", &train_pos_m);
	train_pos.resize(train_pos_m);
	for(int i = 0; i < train_pos_m; i++){
		int x, y;
		scanf("%d%d", &x, &y);
		train_pos[i].x = x;
		train_pos[i].y = y;
		edge[x].push_back(y);
		edge[y].push_back(x);
	}
	scanf("%d", &train_neg_m);
	train_neg.resize(train_neg_m);
	for(int i = 0; i < train_neg_m; i++)
		scanf("%d%d", &train_neg[i].x, &train_neg[i].y);
	scanf("%d", &valid_pos_m);
	valid_pos.resize(valid_pos_m);
	for(int i = 0; i < valid_pos_m; i++)
		scanf("%d%d", &valid_pos[i].x, &valid_pos[i].y);
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
	for(int i = 0; i < num_nodes; i++)vis[i] = -1;
	DSU dsu(num_nodes);
	for(int i = 0; i < train_pos_m;i++)
		dsu.merge(train_pos[i].x, train_pos[i].y);
	for(int i = 0; i < num_nodes;i++)
		if(dsu[i] == dsu[0])vertices1.push_back(i);else vertices2.push_back(i);
	for(int x : vertices){
		int y = vertices1[R(vertices1.size())];
		edge[x].push_back(y);
		edge[y].push_back(x);
	}
	for(int i = 0; i < num; i++){
		cerr<<i<<endl;
		int st = R(num_nodes);
		queue<int>Q;
		for(int j = 0; j < num_nodes; j++)
			d[j] = 100;
		Q.push(st);
		vis[st] = i;
		d[st] = 0;
		while(!Q.empty()){
			int x = Q.front();
			Q.pop();
			for(int y : edge[x])
				if(vis[y] != i){
					vis[y] = i;
					d[y] = d[x] + 1;
					Q.push(y);
				}
		}
		for(int j = 0; j < train_pos_m; j++)
			train_pos[j].z += d[train_pos[j].x] + d[train_pos[j].y];
		for(int j = 0; j < train_neg_m; j++)
			train_neg[j].z += d[train_neg[j].x] + d[train_neg[j].y];
		for(int j = 0; j < valid_pos_m; j++)
			valid_pos[j].z += d[valid_pos[j].x] + d[valid_pos[j].y];
		for(int j = 0; j < valid_neg_m; j++)
			valid_neg[j].z += d[valid_neg[j].x] + d[valid_neg[j].y];
		for(int j = 0; j < test_pos_m; j++)
			test_pos[j].z += d[test_pos[j].x] + d[test_pos[j].y];
		for(int j = 0; j < test_neg_m; j++)
			test_neg[j].z += d[test_neg[j].x] + d[test_neg[j].y];
	}
	freopen((string(out_dir) + string("anchor_distance_train_pos.txt")).c_str(), "w", stdout);
	for(int i = 0; i < train_pos_m; i++)
		printf("%.15lf\n", 1.0 * train_pos[i].z / num);
	fclose(stdout);
	freopen((string(out_dir) + string("anchor_distance_train_neg.txt")).c_str(), "w", stdout);
	for(int i = 0; i < train_neg_m; i++)
		printf("%.15lf\n", 1.0 * train_neg[i].z / num);
	fclose(stdout);
	freopen((string(out_dir) + string("anchor_distance_valid_pos.txt")).c_str(), "w", stdout);
	for(int i = 0; i < valid_pos_m; i++)
		printf("%.15lf\n", 1.0 * valid_pos[i].z / num);
	fclose(stdout);
	freopen((string(out_dir) + string("anchor_distance_valid_neg.txt")).c_str(), "w", stdout);
	for(int i = 0; i < valid_neg_m; i++)
		printf("%.15lf\n", 1.0 * valid_neg[i].z / num);
	fclose(stdout);
	freopen((string(out_dir) + string("anchor_distance_test_pos.txt")).c_str(), "w", stdout);
	for(int i = 0; i < test_pos_m; i++)
		printf("%.15lf\n", 1.0 * test_pos[i].z / num);
	fclose(stdout);
	freopen((string(out_dir) + string("anchor_distance_test_neg.txt")).c_str(), "w", stdout);
	for(int i = 0; i < test_neg_m; i++)
		printf("%.15lf\n", 1.0 * test_neg[i].z / num);
	fclose(stdout);
}
