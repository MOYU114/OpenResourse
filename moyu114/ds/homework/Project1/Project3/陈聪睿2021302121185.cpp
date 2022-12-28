#include<cstdio>
#include<iostream>
#include<stack>
using namespace std;
void DecToOct(int n) 
{
	stack<int>s1;
	int i;
	while (n) 
	{
		i=n % 8;
		n /= 8;
		s1.push(i);
	}
	i = s1.size();
	while (i--) {
		cout << s1.top();
		s1.pop();
	}
	cout<<endl;
}
void DecToHex(int n) 
{
	stack<char>s1;
	char z;
	int i, cnt = 0;
	while (n) 
	{
		i = n % 16;
		n /= 16;
		z = '0'+i;
		switch (i) {
		case 10:
			s1.push('A');
			break;
		case 11:
			s1.push('B');
			break;
		case 12:
			s1.push('C');
			break;
		case 13:
			s1.push('D');
			break;
		case 14:
			s1.push('E');
			break;
		case 15:
			s1.push('F');
			break;
		default:
			s1.push(z);
		}
	}
	i = s1.size();
	while (i--) {
		cout << s1.top();
		s1.pop();
	}
	cout << endl;
}
int main() 
{
	int n;
	cout << "请输入你先转变的十进制数：" ;
	cin >> n;
	cout <<n<< "转变为八进制数：" ;
	DecToOct(n);
	cout << n << "转变为十六进制数：";
	DecToHex(n);
	return 0;
}