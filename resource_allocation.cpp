#include"resource_allocation.hpp"
struct P{
	int x, y;
};
void resource_allocation(char *in_dir, char *out_dir, bool use_val){
	int num_nodes, train_pos_m, train_neg_m, valid_pos_m, valid_neg_m, test_pos_m, test_neg_m;
	vector<P> train_pos;
	vector<P> valid_pos;
	vector<vector<int> > edge;
	vector<int> deg;
	vector<double> score;
	freopen(in_dir, "r", stdin);
	scanf("%d", &num_nodes);
	edge.resize(num_nodes);
	deg.resize(num_nodes);
	score.resize(num_nodes);
	for(int i = 0; i < num_nodes; i++)deg[i] = 0;
	scanf("%d", &train_pos_m);
	train_pos.resize(train_pos_m);
	for(int i = 0; i < train_pos_m; i++){
		int x, y;
		scanf("%d%d", &x, &y);
		train_pos[i].x = x;
		train_pos[i].y = y;
		edge[x].push_back(y);
		edge[y].push_back(x);
		deg[x]++;
		deg[y]++;
	}
	for(int i = 0; i < num_nodes; i++){
		sort(edge[i].begin(), edge[i].end());
		edge[i].erase(unique(edge[i].begin(), edge[i].end()), edge[i].end());
		if(deg[i] > 0)score[i] = 1.0 / sqrt(1.0 * deg[i]);
	}
	freopen((string(out_dir) + string("resource_allocation_train_pos.txt")).c_str(), "w", stdout);
	for(int i = 0; i < train_pos_m; i++){
		int x = train_pos[i].x, y = train_pos[i].y;
		vector<int>::iterator it = edge[y].begin();
		double sum = 0;
		for(int u : edge[x]){
			while(it != edge[y].end() && (*it) < u) it++;
			if(it != edge[y].end() && (*it) == u) sum += score[*it];
		}
		printf("%.12lf\n", sum);
	}
	fclose(stdout);
	freopen((string(out_dir) + string("resource_allocation_train_neg.txt")).c_str(), "w", stdout);
	scanf("%d", &train_neg_m);
	for(int i = 0; i < train_neg_m; i++){
		int x, y;
		scanf("%d%d", &x, &y);
		vector<int>::iterator it = edge[y].begin();
		double sum = 0;
		for(int u : edge[x]){
			while(it != edge[y].end() && (*it) < u) it++;
			if(it != edge[y].end() && (*it) == u) sum += score[*it];
		}
		printf("%.12lf\n", sum);
	}
	fclose(stdout);
	freopen((string(out_dir) + string("resource_allocation_valid_pos.txt")).c_str(), "w", stdout);
	scanf("%d", &valid_pos_m);
	valid_pos.resize(valid_pos_m);
	for(int i = 0; i < valid_pos_m; i++){
		int x, y;
		scanf("%d%d", &x, &y);
		valid_pos[i].x = x;
		valid_pos[i].y = y;
		vector<int>::iterator it = edge[y].begin();
		double sum = 0;
		for(int u : edge[x]){
			while(it != edge[y].end() && (*it) < u) it++;
			if(it != edge[y].end() && (*it) == u) sum += score[*it];
		}
		printf("%.12lf\n", sum);
	}
	fclose(stdout);
	freopen((string(out_dir) + string("resource_allocation_valid_neg.txt")).c_str(), "w", stdout);
	scanf("%d", &valid_neg_m);
	for(int i = 0; i < valid_neg_m; i++){
		int x, y;
		scanf("%d%d", &x, &y);
		vector<int>::iterator it = edge[y].begin();
		double sum = 0;
		for(int u : edge[x]){
			while(it != edge[y].end() && (*it) < u) it++;
			if(it != edge[y].end() && (*it) == u) sum += score[*it];
		}
		printf("%.12lf\n", sum);
	}
	fclose(stdout);
	if(use_val){
		for(P val_edge : valid_pos){
			edge[val_edge.x].push_back(val_edge.y);
			edge[val_edge.y].push_back(val_edge.x);
			deg[val_edge.x]++;
			deg[val_edge.y]++;
		}
		for(int i = 0; i < num_nodes; i++){
			sort(edge[i].begin(), edge[i].end());
			edge[i].erase(unique(edge[i].begin(), edge[i].end()), edge[i].end());
			if(deg[i] > 0)score[i] = 1.0 / sqrt(1.0 * deg[i]);
		}
	}
	freopen((string(out_dir) + string("resource_allocation_test_pos.txt")).c_str(), "w", stdout);
	scanf("%d", &test_pos_m);
	for(int i = 0; i < test_pos_m; i++){
		int x, y;
		scanf("%d%d", &x, &y);
		vector<int>::iterator it = edge[y].begin();
		double sum = 0;
		for(int u : edge[x]){
			while(it != edge[y].end() && (*it) < u) it++;
			if(it != edge[y].end() && (*it) == u) sum += score[*it];
		}
		printf("%.12lf\n", sum);
	}
	fclose(stdout);
	freopen((string(out_dir) + string("resource_allocation_test_neg.txt")).c_str(), "w", stdout);
	scanf("%d", &test_neg_m);
	for(int i = 0; i < test_neg_m; i++){
		int x, y;
		scanf("%d%d", &x, &y);
		vector<int>::iterator it = edge[y].begin();
		double sum = 0;
		for(int u : edge[x]){
			while(it != edge[y].end() && (*it) < u) it++;
			if(it != edge[y].end() && (*it) == u) sum += score[*it];
		}
		printf("%.12lf\n", sum);
	}
	fclose(stdout);
	fclose(stdin);
}
