#include"shortest_path.hpp"
struct P{
	int x, y;
};
void shortest_path(char *in_dir, char *out_dir, int maxv){
	int num_nodes, train_pos_m, train_neg_m, valid_pos_m, valid_neg_m, test_pos_m, test_neg_m;
	vector<P>train_pos;
	vector<vector<int> > edge;
	map<pair<int,int>,int> cnt;
	vector<int> vis, d;
	freopen(in_dir, "r", stdin);
	scanf("%d", &num_nodes);
	edge.resize(num_nodes);
	d.resize(num_nodes);
	vis.resize(num_nodes);
	scanf("%d", &train_pos_m);
	train_pos.resize(train_pos_m);
	for(int i = 0; i < train_pos_m; i++){
		int x, y;
		scanf("%d%d", &x, &y);
		train_pos[i].x = x;
		train_pos[i].y = y;
		edge[x].push_back(y);
		edge[y].push_back(x);
		cnt[make_pair(x, y)]++;
		cnt[make_pair(y, x)]++;
	}
	freopen((string(out_dir) + string("shortest_path_train_pos.txt")).c_str(), "w", stdout);
	for(int i = 0; i < num_nodes; i++)vis[i] = -1;
	for(int i = 0; i < train_pos_m; i++){
		int st = train_pos[i].x, ed = train_pos[i].y;
		queue<int>Q;
		Q.push(st);
		vis[st] = i;
		d[st] = 0;
		while(!Q.empty()){
			int x = Q.front();
			Q.pop();
			for(int y : edge[x])
				if(vis[y] != i){
					if(x == st && y == ed && cnt[make_pair(x, y)] == 1)continue;
					vis[y] = i;
					d[y] = d[x] + 1;
					if(y == ed)break;
					Q.push(y);
				}
		}
		printf("%d\n", vis[ed] == i ? d[ed] : maxv);
	}
	fclose(stdout);
	freopen((string(out_dir) + string("shortest_path_train_neg.txt")).c_str(), "w", stdout);
	for(int i = 0; i < num_nodes; i++)vis[i] = -1;
	scanf("%d", &train_neg_m);
	for(int i = 0; i < train_neg_m; i++){
		int st, ed;
		queue<int>Q;
		scanf("%d%d",&st, &ed);
		Q.push(st);
		vis[st] = i;
		d[st] = 0;
		while(!Q.empty()){
			int x = Q.front();
			Q.pop();
			for(int y : edge[x])
				if(vis[y] != i){
					if(x == st && y == ed && cnt[make_pair(x, y)] == 1)continue;
					vis[y] = i;
					d[y] = d[x] + 1;
					if(y == ed)break;
					Q.push(y);
				}
		}
		printf("%d\n", vis[ed] == i ? d[ed] : maxv);
	}
	fclose(stdout);
	freopen((string(out_dir) + string("shortest_path_valid_pos.txt")).c_str(), "w", stdout);
	for(int i = 0; i < num_nodes; i++)vis[i] = -1;
	scanf("%d", &valid_pos_m);
	for(int i = 0; i < valid_pos_m; i++){
		int st, ed;
		queue<int>Q;
		scanf("%d%d",&st, &ed);
		Q.push(st);
		vis[st] = i;
		d[st] = 0;
		while(!Q.empty()){
			int x = Q.front();
			Q.pop();
			for(int y : edge[x])
				if(vis[y] != i){
					if(x == st && y == ed && cnt[make_pair(x, y)] == 1)continue;
					vis[y] = i;
					d[y] = d[x] + 1;
					if(y == ed)break;
					Q.push(y);
				}
		}
		printf("%d\n", vis[ed] == i ? d[ed] : maxv);
	}
	fclose(stdout);
	freopen((string(out_dir) + string("shortest_path_valid_neg.txt")).c_str(), "w", stdout);
	for(int i = 0; i < num_nodes; i++)vis[i] = -1;
	scanf("%d", &valid_neg_m);
	for(int i = 0; i < valid_neg_m; i++){
		int st, ed;
		queue<int>Q;
		scanf("%d%d",&st, &ed);
		Q.push(st);
		vis[st] = i;
		d[st] = 0;
		while(!Q.empty()){
			int x = Q.front();
			Q.pop();
			for(int y : edge[x])
				if(vis[y] != i){
					if(x == st && y == ed && cnt[make_pair(x, y)] == 1)continue;
					vis[y] = i;
					d[y] = d[x] + 1;
					if(y == ed)break;
					Q.push(y);
				}
		}
		printf("%d\n", vis[ed] == i ? d[ed] : maxv);
	}
	fclose(stdout);
	freopen((string(out_dir) + string("shortest_path_test_pos.txt")).c_str(), "w", stdout);
	for(int i = 0; i < num_nodes; i++)vis[i] = -1;
	scanf("%d", &test_pos_m);
	for(int i = 0; i < test_pos_m; i++){
		int st, ed;
		queue<int>Q;
		scanf("%d%d",&st, &ed);
		Q.push(st);
		vis[st] = i;
		d[st] = 0;
		while(!Q.empty()){
			int x=Q.front();
			Q.pop();
			for(int y : edge[x])
				if(vis[y] != i){
					if(x == st && y == ed && cnt[make_pair(x, y)] == 1)continue;
					vis[y] = i;
					d[y] = d[x] + 1;
					if(y == ed)break;
					Q.push(y);
				}
		}
		printf("%d\n", vis[ed] == i ? d[ed] : maxv);
	}
	fclose(stdout);
	freopen((string(out_dir) + string("shortest_path_test_neg.txt")).c_str(), "w", stdout);
	for(int i = 0; i < num_nodes; i++)vis[i] = -1;
	scanf("%d", &test_neg_m);
	for(int i = 0; i < test_neg_m; i++){
		int st, ed;
		queue<int>Q;
		scanf("%d%d",&st, &ed);
		Q.push(st);
		vis[st] = i;
		d[st] = 0;
		while(!Q.empty()){
			int x=Q.front();
			Q.pop();
			for(int y : edge[x])
				if(vis[y] != i){
					if(x == st && y == ed && cnt[make_pair(x, y)] == 1)continue;
					vis[y] = i;
					d[y] = d[x] + 1;
					if(y == ed)break;
					Q.push(y);
				}
		}
		printf("%d\n", vis[ed] == i ? d[ed] : maxv);
	}
	fclose(stdout);
	fclose(stdin);
}
