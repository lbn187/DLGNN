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
			seq[pos[x] = mt++] = make_pair(d[x], x);
			for(auto y:edge[x])
				if(y != fa){
					dfs(y, x);
					seq[mt++] = make_pair(d[x], x);
				}
		}
		void init(){
			flag=true;
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
		~Tree(){
			for(vector<int> x:edge){
				x.clear();
				vector<int>(x).swap(x);
			}
			edge.clear();
			vector<vector<int> >(edge).swap(edge);
			seq.clear();
			vector<pair<int,int> >(seq).swap(seq);
			d.clear();
			vector<int>(d).swap(d);
			pos.clear();
			vector<int>(pos).swap(pos);
		}
		Tree(int _n):n(_n),rt(0),flag(false){
			edge.resize(n);
		}
		void add_edge(int x, int y){
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
};
