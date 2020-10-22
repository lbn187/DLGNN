#include<bits/stdc++.h>
#include "rmq.hpp"
using namespace std;
struct Tree{
	private:
		int n, rt, mt;
		bool flag;
		vector<vector<int> >edge;
		RMQ<pair<int,int> > rmq;
		vector<pair<int,int> > seq;
		vector<int> d, pos;
		void dfsdeep(int x, int fa, int dis){
			d[x] = dis;
			for(int y:edge[x])
				if(y!=fa)
					dfsdeep(y, x, dis+1);
		}
		void dfs(int x, int fa){
			seq[pos[x] = ++mt] = make_pair(d[x], x);
			for(auto y:edge[x])
				if(y != fa){
					dfs(y, x);
					seq[++mt] = make_pair(d[x], x);
				}
		}
		void init(){
			d.resize(n);
			dfsdeep(rt, -1, 1);
			mt = 0;
			pos.resize(n);
			seq.resize(n * 2 - 1);
			dfs(rt, 0);
			rmq.init(seq);
		}
	public:
		Tree()=default;
		Tree(int _n):n(_n),rt(0),flag(false){
			edge.resize(n);
		}
		void ins(int x, int y){
			edge[x].push_back(y);
			edge[y].push_back(x);
		}
		int lca(int x, int y){
			if(!flag)init();
			if(pos[x] > pos[y])swap(x, y);
			return rmq.get(pos[x], pos[y]).second;
		}
		int dis(int x, int y){
			int z = lca(x, y);
			return d[x] + d[y] - 2 * d[z];
		}
}