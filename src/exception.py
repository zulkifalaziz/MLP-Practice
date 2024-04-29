import logging
import  sys
import logger
def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    error_message = (f'error occured in python script name [{exc_tb.tb_frame.f_code.co_filename}] '
                     f'line number [{exc_tb.tb_lineno}] '
                     f'error message [{str(error)}]')

    return error_message
    pass


class CustomError(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error=error_message, error_detail=error_detail)
        pass

    def __str__(self):
        return self.error_message
        pass
    pass


if __name__ == '__main__':
    try:
        print(1/0)
    except Exception as e:
        logging.info('Division by Zero Error')
        raise CustomError(e, sys)