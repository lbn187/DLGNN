#include<bits/stdc++.h>
using namespace std;
struct P{
	int x, y;
};
void katz_distance(char *in_dir, char *out_dir, double beta, int max_length){
	int num_nodes, train_pos_m, train_neg_m, valid_pos_m, valid_neg_m, test_pos_m, test_neg_m;
	vector<P> train_pos, train_pos2, train_neg, valid_pos, valid_neg, test_pos, test_neg;
	vector<vector<int> >edge;
	vector<vector<double> >dp, dp_nxt, score;
	freopen(in_dir, "r", stdin);
	scanf("%d", &num_nodes);
	edge.resize(num_nodes);
	dp.resize(num_nodes);
	dp_nxt.resize(num_nodes);
	score.resize(num_nodes);
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
	fclose(stdin);
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
	for(int i = 0; i < num_nodes; i++){
		dp[i].resize(num_nodes);
		dp_nxt[i].resize(num_nodes);
		dp[i][i] = 1.0;
	}
	double tmp = 1.0;
	for(step = 1; step < max_length; step++){
		tmp = tmp * beta;
		for(int i = 0; i < num_nodes; i++)
			for(int j = 0; j < num_nodes; j++)
				for(int x : edge[j])
					dp_nxt[i][x] += dp[i][j];
		for(int i = 0; i < num_nodes; i++)
			for(int j = 0; j < num_nodes; j++){
				dp[i][j] = dp_nxt[i][j];
				dp_nxt[i][j] = 0;
				score[i][j] += tmp * dp[i][j];
			}
	}
	freopen((string(out_dir) + string("train_pos_katz_distance.txt")).c_str(), "w", stdout);
	for(int i = 0; i < train_pos_m; i++)
		printf("%.15lf\n", score[train_pos[i].x][train_pos[i].y]);
	fclose(stdout);
	freopen((string(out_dir) + string("train_neg_katz_distance.txt")).c_str(), "w", stdout);
	for(int i = 0; i < train_neg_m; i++)
		printf("%.15lf\n", score[train_neg[i].x][train_neg[i].y]);
	fclose(stdout);
	freopen((string(out_dir) + string("valid_pos_katz_distance.txt")).c_str(), "w", stdout);
	for(int i = 0; i < valid_pos_m; i++)
		printf("%.15lf\n", score[valid_pos[i].x][valid_pos[i].y]);
	fclose(stdout);
	freopen((string(out_dir) + string("valid_neg_katz_distance.txt")).c_str(), "w", stdout);
	for(int i = 0; i < valid_neg_m; i++)
		printf("%.15lf\n", score[valid_neg[i].x][valid_neg[i].y]);
	fclose(stdout);
	freopen((string(out_dir) + string("test_pos_katz_distance.txt")).c_str(), "w", stdout);
	for(int i = 0; i < test_pos_m; i++)
		printf("%.15lf\n", score[test_pos[i].x][test_pos[i].y]);
	fclose(stdout);
	freopen((string(out_dir) + string("test_neg_katz_distance.txt")).c_str(), "w", stdout);
	for(int i = 0; i < test_neg_m; i++)
		printf("%.15lf\n", score[test_neg[i].x][test_neg[i].y]);
	fclose(stdout);
}
