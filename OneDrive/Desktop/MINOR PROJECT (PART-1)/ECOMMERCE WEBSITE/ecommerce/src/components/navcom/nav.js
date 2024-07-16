import React, { useContext, useState } from "react";
import { Link } from "react-router-dom";
import "./nav.css";
import logo from "../images/Assets/online-shop.jpg";
import cart_icon from "../images/Assets/cart_icon.png";
import { ShopContext } from "../../context/shopcontext";

export default function Nav() {
  const [menu, setmenu] = useState("shop");
  const {  getTotalCartItems}=useContext(ShopContext);
  return (
    <div className="main_nav">
      <div className="nav_bar">
        <div className="nav-logo">
          <img src={logo} alt=""></img>
          <p>SHOPIFY</p>
        </div>

        <ul className="nav-menu">
        <li
            onClick={() => {
              setmenu("shop");
            }}
          >
            {/* Shop */}

            <Link style={{textDecoration:'none'}}  to="/">Shop</Link>
            {menu === "shop" ? <hr /> : <></>}
          </li>
          <li
            onClick={() => {
              setmenu("men");
            }}
          >
            {/* <a href="/mens" target="_blank">  mens</a> */}
            <Link style={{textDecoration:'none'}} to="/mens">Men</Link>

            {menu === "men" ? <hr /> : <></>}
          </li>
          <li
            onClick={() => {
              setmenu("women");
            }}
          >
            <Link style={{textDecoration:'none'}}  to="/womens">Women</Link>

            {menu === "women" ? <hr /> : <></>}
          </li>
          <li
            onClick={() => {
              setmenu("kids");
            }}
          >
            <Link style={{textDecoration:'none'}}  to="/kids">Kids</Link>

            {menu === "kids" ? <hr /> : <></>}
          </li>
        </ul>
        <div className="nav-login-cart">
          <Link to='/login'>
          <button>Login</button>
          </Link>
          <Link to='/cart'>
          <img src={cart_icon}></img>
          </Link>
          <div className="nav-cart-count">{  getTotalCartItems()}</div>
        </div>
      </div>
      <hr></hr>
    </div>
  );
}
