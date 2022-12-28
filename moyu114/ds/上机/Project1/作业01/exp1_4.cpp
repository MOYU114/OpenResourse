#include<cstdio>
#include<iostream>
using namespace std;
long long fun(double n) {
    long long result=0,temp=1;//result存储最终结果，temp存储每次的n！

     for (int i = 1;i <= n;i++) {//只使用一次循环，时间复杂度为O（n）
         temp *= i;
         result = result + temp;
    }
     return result;
}
 int main() {
	 int n;
	 cin >> n;
	 cout << "Σ(k=1,n)k!=" << fun(n) << endl;
 }