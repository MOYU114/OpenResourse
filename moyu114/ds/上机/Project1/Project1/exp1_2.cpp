#include<cstdio>
#include<iostream>
#include<math.h>
using namespace std;
void fun(int a[], double n) {
	int i, j;
	int p, h;
	a[1] = 1;
	p = 1;
	for (i = 2;i <= n;i++) {
		for (j = 1, h = 0;j <= p;j++) {
			a[j] = a[j] * i + h;
			h = a[j] / 10;
			a[j] = a[j] % 10;
		}
		while (h > 0) {
			a[j] = h % 10;
			h /= 10;
			j++;
		}
		p = j - 1;
	}
		for (i = p;i >= 2;i--) {
			printf("%d", a[i]);
		}
		printf("%d", a[i]);
}
int main() {
	double n, i;
	cin >> n;
    int a[50000] = { 1 };
	for (i = 1;i <= n;i++) {
		cout << "log2("<<i<<")=" << log2(i) << endl;
		cout << "sqrt(" << i << ")=" << sqrt(i) << endl;
		cout << i << "=" << i << endl;
		cout << i << "log2(" << i << ")=" << i * log2(i) << endl;
		cout << i << "^2=" << pow(i, 2) << endl;
		cout << i<<" ^3=" << pow(i, 3) << endl;
		cout << "2^" << i << "=" << (long long)pow(2, i) << endl;
        cout << i << " !=";fun(a, i);
		cout << endl;
	}
}