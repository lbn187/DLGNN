#include<bits/stdc++.h>
using namespace std;
template<typename T> struct RMQ{
	private:
		int n, high;
		vector<vector<T> >f;
	public:
		RMQ()=default;
		~RMQ(){
			for(vector<T> x:f){
				x.clear();
				vector<T>(x).swap(x);
			}
			f.clear();
			vector<vector<T> >(f).swap(f);
		}
		void init(vector<T> v){
			n=(int)v.size();
			f.resize(n);
			high = 32 - __builtin_clz(n);
			for(int i = 0; i < n; i++)
				f[i].push_back(v[i]);
			for(int j = 1; j <= high; j++)
				for(int i = 0; i < n - (1 << j) + 1; i++)
					f[i].push_back(min(f[i][j - 1], f[i + (1 << j - 1)][j - 1]));
		}
		T get(int l, int r){
			int t = 31 - __builtin_clz(r - l + 1);
			return min(f[l][t], f[r - (1 << t) + 1][t]);
		}
};
