#include<cstdio>
#include<iostream>
#include<vector>
using namespace std;
int prime1(int n) {
	int cnt=1;          //将素数2包含进去
	for(int i = 2;i <=n;i++) {
		for (int j = 2;j <i;j++) {
			if (i % j == 0) {
				cnt++;
				break;
			}
		}
	}
	return n-cnt;
}
int prime2(int n) {
	vector<int>a(n+1,1);
	int cnt = 0;
	int i, j;
	for (i = 2;i <= n;i++) {
		if (a.at(i) == 1) {
			cnt++;
			for (j = 2;j * i <= n;j++) {
				a.at(i * j) = 0;
			}
		}
	}
	return cnt;
}

int main() {
	
	int n;
	cin >> n;
	
	cout << prime1(n) << endl;//1~100内，debug模式下，用时1.947s
	cout << prime2(n) << endl;//1~100内，debug模式下，用时1.437s
}


