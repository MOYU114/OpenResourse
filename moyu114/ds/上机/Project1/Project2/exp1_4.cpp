#include<cstdio>
#include<iostream>
using namespace std;
long long fun(double n) {
    long long result=0,temp=1;//result�洢���ս����temp�洢ÿ�ε�n��

     for (int i = 1;i <= n;i++) {//ֻʹ��һ��ѭ����ʱ�临�Ӷ�ΪO��n��
         temp *= i;
         result = result + temp;
    }
     return result;
}
 int main() {
	 int n;
	 cin >> n;
	 cout << "��(k=1,n)k!=" << fun(n) << endl;
 }