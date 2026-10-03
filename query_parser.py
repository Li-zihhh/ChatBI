

class QueryParser:
    """
    用户查询解析器
    """

    def parse(self, user_question: str) -> dict:
        """ 解析用户输入的查询"""
        return {
            "original_question": user_question.strip(),
            "is_valid": len(user_question.strip()) > 0
        }

    def validate(self, parsed_query: dict) -> bool:
        """ 验证解析后的查询是否有效"""
        return parsed_query.get("is_valid", False)













