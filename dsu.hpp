#include<bits/stdc++.h>
struct DSU{
	private:
		int n;
		vector<int> father;
	public:
		DSU()=default;
		DSU(int _n):n(_n){init(n);}
		void init(int _n){
			n = _n;
			father.resize(n);
			for(int i = 0; i < n; i++)
				father[i] = i;
		}
		int find(int x){
			return father[x] == x ? x : father[x] = find(father[x]);
		}
		int& operator[](int x){
			return F[find(x)];
		}
		bool merge(int x, int y){
			x = find(x);
			y = find(y);
			if(x == y)return false;
			father[x] = y;
			return true;
		}
}