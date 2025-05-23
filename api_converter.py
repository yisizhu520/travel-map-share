import json
import re
import requests
import argparse

def get_api_detail(api_id, headers=None, cookies=None):
    """根据api_id获取API详情"""
    url = f'https://open-platform.qunhequnhe.com/docpub/api/openapi/detail?api_id={api_id}'
    
    if headers is None:
        headers = {
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
            'cache-control': 'no-cache',
            'pragma': 'no-cache',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
        }
    
    response = requests.get(url, headers=headers, cookies=cookies)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"获取API详情失败，状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        return None

def extract_curl_headers(file_path):
    """从curl命令中提取headers和cookies"""
    headers = {}
    cookies = {}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 提取headers
    header_matches = re.findall(r'-H \'([^:]+): ([^\']*)\'', content)
    for key, value in header_matches:
        headers[key] = value
    
    # 提取cookies
    cookie_match = re.search(r'-b \'([^\']*)\'', content)
    if cookie_match:
        cookie_str = cookie_match.group(1)
        cookie_pairs = cookie_str.split('; ')
        for pair in cookie_pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                cookies[key] = value
    
    return headers, cookies

def map_detail_to_create(detail_data):
    """将detail.txt的响应数据映射到create.txt的请求体格式"""
    d = detail_data.get('d', {})
    
    # 构建请求体
    create_body = {
        "apiType": 0,
        "apiMethod": d.get("apiMethod", "POST"),
        "serviceType": d.get("serviceType", 0),
        "ownerGroup": d.get("ownerGroup", "all"),
        "apiName": d.get("apiName", ""),
        "description": d.get("description", ""),
        "tester": d.get("tester", ""),
        "regionInfos": [
            {
                "region": region.get("region", 0),
                "url": region.get("url", "").replace("/oauth2/openapi/v1/", "/p/openapi/v2/"),
                "prefix": "/p/openapi/v2/",
                "defaultParamMappingSwitch": True
            } for region in d.get("regionInfos", [])
        ],
        "serviceApiUrl": d.get("serviceApiUrl", ""),
        "serviceApiMethod": d.get("serviceApiMethod", "POST"),
        "apiSecurityLevel": 1,  # 从detail中的0改为1
        "responseParamMappingSwitch": False,  # 从detail中的false改为true
        "paramInput": {
            "requestQuery": [
                {
                    "param": "appuid",
                    "required": True,
                    "type": "string",
                    "description": "第三方用户的ID"
                }
            ]
        },
        "returnResult": {
            "param": d.get("returnResult", {}).get("param", "root"),
            "mappedParam": d.get("returnResult", {}).get("mappedParam", "root"),
            "type": "boolean"  # 从detail中的string改为boolean
        },
        "moduleId": 6,  # 从detail中的217改为6
        "authType": 1,  # 从detail中的0改为1
        "serviceHost": d.get("serviceHost", "")
    }
    
    return create_body

def send_create_request(create_body, headers, cookies):
    """发送创建API的请求"""
    url = 'https://open-platform.qunhequnhe.com/docpub/api/openapi/creation'
    
    # 移除不需要的headers
    if 'content-length' in headers:
        del headers['content-length']
    
    # 确保content-type正确
    headers['content-type'] = 'application/json;charset=UTF-8'
    
    response = requests.post(
        url,
        json=create_body,
        headers=headers,
        cookies=cookies
    )
    
    return response

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='根据API ID获取详情并创建新API')
    parser.add_argument('api_id', type=int, help='要获取详情的API ID')
    parser.add_argument('--template', type=str, default=r'c:\workspace\private\travel-map-share\detail.txt',
                        help='包含curl命令的模板文件路径')
    parser.add_argument('--output', type=str, help='输出结果到文件')
    args = parser.parse_args()
    
    # 提取headers和cookies
    headers, cookies = extract_curl_headers(args.template)
    
    # 获取API详情
    print(f"正在获取API ID {args.api_id} 的详情...")
    detail_data = get_api_detail(args.api_id, headers, cookies)
    
    if not detail_data:
        print("无法获取API详情数据")
        return
    
    # 打印获取到的API详情
    print(f"成功获取API '{detail_data.get('d', {}).get('apiName', '')}' 的详情")
    
    # 映射数据
    create_body = map_detail_to_create(detail_data)
    
    # 打印请求体以便检查
    print("生成的请求体:")
    create_body_json = json.dumps(create_body, indent=2, ensure_ascii=False)
    print(create_body_json)
    
    # 保存到输出文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(create_body_json)
        print(f"请求体已保存到 {args.output}")
    
    # 询问是否发送请求
    send_request = input("是否发送创建API的请求? (y/n): ").lower().strip() == 'y'
    
    if send_request:
        try:
            response = send_create_request(create_body, headers, cookies)
            print(f"请求状态码: {response.status_code}")
            print("响应内容:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"发送请求时出错: {e}")

if __name__ == "__main__":
    main()