from lark import Lark, Token
from lark import UnexpectedToken, UnexpectedCharacters
from lark.lexer import PatternRE, PatternStr
from lark.reconstruct import Reconstructor
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark import GrammarError
from typing import Union, Callable, Optional


def is_pattern_regex(pattern: Union[PatternRE, PatternStr]):
        return isinstance(pattern, PatternRE)
    
def is_pattern_string(pattern: Union[PatternRE, PatternStr]):
    return isinstance(pattern, PatternStr)


class Rebuilder:

    def __init__(self, 
                 grammar: str, 
                 term_subs: dict[str, str] = None,
                 token_transformer: Callable[[str, str, bool], str] = None,
                 ):
        """
        term_subs: dict[str, str] = None, -> term subs when reconstructing grammar from ast
        token_transformer: Callable[[str, str, bool], str] = None, -> token_name, inputs string and is_regex flag, returns token defs
        """
        self.grammar = grammar
        self.parser = Lark(grammar, 
                           parser="lalr", 
                           start="start", 
                           strict=False, 
                           lexer="contextual", 
                           maybe_placeholders=False,
                           regex=True
                           )
        self.terminals = self.parser.lexer_conf.terminals_by_name
        self.reconstructor = Reconstructor(self.parser, term_subs) \
                                if term_subs else Reconstructor(self.parser)
        self.token_transformer = token_transformer if token_transformer else None
    
    def get_token_definition(self, token: str):
        pattern = self.terminals.get(token, None)
        if pattern is not None:
            return pattern.pattern
        return None
    
    def as_token(self, token: str) -> Token:
        tok_def = self.get_token_definition(token)
        if self.token_transformer:
            tok_def = self.token_transformer(token, tok_def, is_pattern_regex(tok_def))
        if tok_def is None:
            tok_def = token
        return Token(token, tok_def)
    
    def beam_search(self, 
                    parser: InteractiveParser, 
                    given_token: Token, 
                    beam_width: int = 10, 
                    strategy: str = "shortest", # longest
                    break_early_limit: Optional[int] = 3,
                    ) -> list[Token]:
        """
        strategy: "shortest" or "longest"
        break_early_limit: 
        if strategy is "shortest", break early as soon as one of any path 
        that consist no more than break_early_limit tokens is found, priorisizing the shortest path
        if strategy is "longest", break early as soon as lenght of tokens path is at least break_early_limit,
        priorisizing the longest path


        !NOTE: longest strategy is buggy for now, we are working on it
        """
        assert strategy in ["shortest", "longest"], "strategy must be either 'shortest' or 'longest'"
        if break_early_limit is not None:
            assert break_early_limit <= beam_width, "break_early_limit must be less than or equal to beam_width"
        if beam_width <= 0:
            return []
        candidates = {} # token2shortest_path
        candidates_score = {} # score2token
        accepted_tokens = list(parser.accepts()) # returns list of token names
        accepted_tokens.sort(key=lambda x: int(is_pattern_string(x)), reverse=False) # first -> string, then -> regex
        for token in accepted_tokens:
            dummy_parser = parser.copy()
            token = self.as_token(token)
            try:
                dummy_parser.feed_token(token)
                success = False
                # for the first two cases let's assume that there are missing tokens between tokena and given token
                if given_token.type in dummy_parser.accepts():
                    # case 1: some token is missing
                    candidates[token] = [token, given_token]
                    candidates_score[1] = token
                    success = True
                else:
                    # case 2: a series of tokens are missing
                    res = [token] + self.beam_search(
                        parser=dummy_parser, 
                        given_token=given_token, 
                        beam_width=beam_width-1,
                        strategy=strategy,
                        break_early_limit=None
                    )
                    score = len(res)
                    candidates[token] = res
                    candidates_score[score] = token
                    success = len(res) > 1
                if not success:
                    # if our hypothesis is wrong (no path discovered), then simply chose the correct token and ignore given token
                    # case 3: given token is not needed at all, remove given token from path
                    candidates[token] = [token]
                    candidates_score[0] = token
            except GrammarError as e:
                raise e

            if break_early_limit is None:
                continue
            if strategy == "shortest":
                for i in range(break_early_limit):
                    if i in candidates_score:
                        return candidates[candidates_score[i]] # break early
            elif strategy == "longest":
                for i in range(beam_width, break_early_limit, -1):
                    if i in candidates_score:
                        return candidates[candidates_score[i]] # break early

        if not candidates:
            return []
        if strategy == "shortest":
            best_score = min(candidates_score.keys()) # the shortest path
        elif strategy == "longest":
            best_score = max(candidates_score.keys()) # the longest path
        return candidates[candidates_score[best_score]]
    
    def repair(self, text: str, beam_width: int = 10, strategy: str = "shortest", break_early_limit: Optional[int] = 3):
        def _repair(e):
            if isinstance(e, UnexpectedToken):
                path = self.beam_search(e.interactive_parser, e.token, beam_width, strategy, break_early_limit)
                if not path:
                    raise e
                for token in path:
                    if token.type != "$END":
                        e.interactive_parser.feed_token(token)
                return True
            elif isinstance(e, UnexpectedCharacters):
                return True # simply ignore
            else:
                raise e
        
        tree = self.parser.parse(text, on_error=_repair)
        return self.reconstructor.reconstruct(tree)







