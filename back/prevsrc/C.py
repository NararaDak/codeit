
# 문제
class Book:
    def __init__(self, title, author, year, isbn):
        self.title = title
        self.author = author
        self.year = year
        self.isbn = isbn
        self.is_rented = False  


class Member:
    def __init__(self, name):
        self.name = name
        self.rented_books = []

class Rental:
    def __init__(self, book, member):
        self.book = book
        self.member = member

class LibraryManagement:
    def __init__(self):
        self.books = []
        self.members = []
        self.rentals = []
    
    def add_book(self, title, author, year, isbn):
        new_book = Book(title, author, year, isbn)
        self.books.append(new_book)
        print(f"'{title}'(저자: {author}, 출판년도:{year}) 도서가 추가되었습니다.")
    def print_books(self):
        print("도서 목록:")
        for book in self.books:
            print(f"- '{book.title}'(저자: {book.author}, 출판년도:{book.year})")
    def add_member(self, name):
        new_member = Member(name)
        self.members.append(new_member)
        print(f"회원'{name}'님이 등록되었습니다")
    
    def print_members(self):
        print("회원 목록:")
        for member in self.members:
            rented_titles = [book.title for book in member.rented_books]
            print(f"- {member.name} (대여 중인 도서: {', '.join(rented_titles) if rented_titles else '없음'})")
    def rent_book(self, isbn, member_name):
        book = next((b for b in self.books if b.isbn == isbn), None)
        member = next((m for m in self.members if m.name == member_name), None)
        
        if not book:
            print("해당 ISBN의 도서를 찾을 수 없습니다.")
            return
        if not member:
            print("해당 이름의 회원을 찾을 수 없습니다.")
            return
        if book.is_rented:
            print(f"'{book.title}' 도서는 이미 대여 중입니다.")
            return
        
        book.is_rented = True
        member.rented_books.append(book)
        new_rental = Rental(book, member)
        self.rentals.append(new_rental)
        print(f"'{member.name}' 회원님이 '{book.title}' 도서를 대여하였습니다.")
    
    def return_book(self, isbn, member_name):
        book = next((b for b in self.books if b.isbn == isbn), None)
        member = next((m for m in self.members if m.name == member_name), None)
        if not book:
            print("해당 ISBN의 도서를 찾을 수 없습니다.")
            return
        if not member:
            print("해당 이름의 회원을 찾을 수 없습니다.")
            return
        if book not in member.rented_books:
            print(f"'{member.name}'님은 '{book.title}' 도서를 대여하지 않았습니다.")
            return
        book.is_rented = False
        member.rented_books.remove(book)
        self.rentals = [r for r in self.rentals if r.book != book or r.member != member]
        print(f"'{member.name}' 회원님이 '{book.title}' 도서를 반납하였습니다.")


library = LibraryManagement()
library.add_book("1984", "조지 오웰", 1949, "978-0451524935")
library.add_book("앵무새 죽이기", "하퍼 리", 1960, "978-0446310789")
library.add_member("홍길동")

library.rent_book("978-0451524935", "홍길동")
library.print_books()
library.print_members()
library.return_book("978-0451524935", "홍길동")
library.print_members()