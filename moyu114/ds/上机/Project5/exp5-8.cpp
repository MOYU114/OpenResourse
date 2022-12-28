#include<iostream>
#define MAXN 20
using namespace std;
int maxv;
int maxw;
int x[MAXN];
int W = 7;
int n = 4;
int w[] = { 5,3,2,1 };
int v[] = { 4,4,3,1 };
void knap(int i, int tw, int tv, int op[]) {
	int j;
	if (i >= n) {
		if (tw <= W && tv > maxv) {
			maxv = tv;
			maxw = tw;
			for (j = 1;j <= n;j++)
				x[j] = op[j];
		}
	}
	else {
		op[i] = 1;
		knap(i + 1, tw + w[i], tv + v[i], op);
		op[i] = 0;
		knap(i + 1, tw, tv, op);
	}
}
void dispasolution(int x[], int n) {
	int i;
	cout << "最佳解决方法为：" << endl;
	for (i = 1;i <= n;i++)
		if (x[i] == 1)
			cout << "选取第" << i << "个商品" << endl;
	cout << "总重量为：" << maxw;
	cout << "总价格为：" << maxv;

}
int main() {
	int op[MAXN];
	knap(0, 0, 0, op);
	dispasolution(x, n);

}