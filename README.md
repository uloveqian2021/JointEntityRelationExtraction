### 实体关系抽取(AI开放平台计划任务)
信息抽取(Information Extraction, IE)是从自然语言文本中抽取实体、属性、关系及事件等事实类信息的文本处理技术，
是信息检索、智能问答、智能对话等人工智能应用的重要基础，一直受到业界的广泛关注。信息抽取任务涉及命名实体识别、指代消解、关系分类等复杂技术，极具挑战性。

# 简介
从文本中抽取出实体、实体类型、以及实体之间的关系，一般称为SPO(S:主语, P:宾语, O:谓语)三元组抽取
其中主语和宾语为两个实体，宾语为两个实体之间的关系

输入：《隐秘而伟大》是由王伟执导，李易峰、金晨、王泷正、牛骏峰领衔主演，王小毅、李强等主演的年代剧。
输出：{'spo_list': [{'source': '李易峰', 'source_type': '人物', 'name': '主演', 'target': '隐秘而伟大', 'target_type': '影视作品'},
      {'source': '李强', 'source_type': '人物', 'name': '主演', 'target': '隐秘而伟大', 'target_type': '影视作品'}, 
      {'source': '王伟', 'source_type': '人物', 'name': '导演', 'target': '隐秘而伟大', 'target_type': '影视作品'}, 
      {'source': '牛骏峰', 'source_type': '人物', 'name': '主演', 'target': '隐秘而伟大', 'target_type': '影视作品'}, 
      {'source': '金晨', 'source_type': '人物', 'name': '主演', 'target': '隐秘而伟大', 'target_type': '影视作品'}, 
      {'source': '王泷正', 'source_type': '人物', 'name': '主演', 'target': '隐秘而伟大', 'target_type': '影视作品'},
      {'source': '王小毅', 'source_type': '人物', 'name': '主演', 'target': '隐秘而伟大', 'target_type': '影视作品'}]}

# 开放日志
2020/12/4 完成第一版线上部署代码

# 依赖

tensorflow-gpu==1.14.0
tensorflow==1.14.0
