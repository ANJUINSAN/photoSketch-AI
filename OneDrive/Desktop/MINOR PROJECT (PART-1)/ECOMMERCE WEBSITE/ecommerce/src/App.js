import "./App.css";
import React from "react";

// import { Router,Route,Routes } from 'react-router-dom';
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";

import Nav from "./components/navcom/nav";
import Shop from "./pages/shop";
import Shopcategory from "./pages/shopcategory";
import Product from "./pages/product";
import Cart from "./pages/cart";
import Loginsighup from "./pages/loginsighup";
import Footer from "./components/footer/footer";
import men_banner from "./components/images/Assets/banner_mens.png";
import women_banner from "./components/images/Assets/banner_women.png";
import kid_banner from "./components/images/Assets/banner_kids.png";
import Payments from "./components/payments/payments";
// import "@fortawesome/fontawesome-free/css/all.min.css";

function App() {
  return (
    <div>
      <Router>
        <Nav></Nav>
        <Routes>
          <Route path="/" element={<Shop />}></Route>
          <Route
            path="/mens"
            element={
              <Shopcategory banner={men_banner} category="men"></Shopcategory>
            }
          ></Route>
          <Route
            path="/womens"
            element={
              <Shopcategory
                banner={women_banner}
                category="women"
              ></Shopcategory>
            }
          ></Route>
          <Route
            path="/kids"
            element={
              <Shopcategory banner={kid_banner} category="kid"></Shopcategory>
            }
          ></Route>
          <Route path="/product" element={<Product></Product>}>
            <Route path=":productId" element={<Product></Product>}></Route>
          </Route>
          <Route path="/login" element={<Loginsighup></Loginsighup>}></Route>
          <Route path="/cart" element={<Cart></Cart>}></Route>
          <Route path="/payments" element={<Payments></Payments>}></Route>
        </Routes>
        <Footer></Footer>
      </Router>
    </div>
  );
}

export default App;
